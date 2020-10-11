import torch
from sklearn.cluster import KMeans
import scipy.signal as sgnt
from scipy.sparse import csc_matrix, csr_matrix
import numpy as np


def apply_weight_sharing(mat, bits=12):
    """
    Applies weight sharing to the given model
    """
    data = mat
    if data.numel() < 2**bits:
        return mat
    weight = data.cpu().numpy()
    shape = weight.shape
    #        print(shape)
    matc = weight.reshape(-1,1)
    matc = csc_matrix(matc)
    min_ = min(matc.data)
    max_ = max(matc.data)
    space = np.linspace(min_, max_, num=2**bits)
    try:
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(matc.data.reshape(-1,1))
    except Exception:
        return mat
    new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
    matc.data = new_weight
    mat_n = matc.toarray()
    return mat_n


def to_sparse_encoded_tupple(mat):
    orig_shape = mat.shape
    mat_n = apply_weight_sharing(mat, bits=8)
    mat_n = torch.tensor(mat_n).view(orig_shape)
    new_mat = {}
    new_mat['ind'] = []
    new_mat['values'] = torch.unique(mat_n)
    new_mat['labels_'] = []
    new_mat['orig_shape'] = orig_shape
    mapping = prepare_mapping(new_mat['values'].tolist())
    for i,value in enumerate(mat_n.view(-1)):
        if value == 0:
            continue
        new_mat['ind'].append(int(i))
        new_mat['labels_'].append(int(mapping['{}'.format(value)]))
    return new_mat

def prepare_mapping(values):
    dicti = {}
    for i,v in enumerate(values):
        dicti['{}'.format(v)] = i
    return dicti

def to_dense(mat_dict):
    if type(mat_dict) is not dict:
        return mat_dict
    new_mat = torch.zeros(mat_dict['orig_shape'][0]*mat_dict['orig_shape'][1])
    for cnt,i in enumerate(mat_dict['ind']):
        new_mat[i] = mat_dict['values'][mat_dict['labels_'][cnt]]
    new_mat = new_mat.view(mat_dict['orig_shape'])
    return new_mat


def to_sparse_dict(d,main_field):
    keys = [key for key in d.keys() if main_field in key]
    ln = len(keys)
    cnt = 1
    for key in keys:
        if 'bias' in key:
            continue
        print('progress {} %'.format(200*cnt/ln))
        data = d[key].data
        cnt += 1
        if len(data.size()) == 1 :
            continue
        sparsed_struct = to_sparse_encoded_tupple(data)
        d[key] = sparsed_struct
    return d

def to_dense_dict(d,main_field):
    keys = [key for key in d.keys() if main_field in key]
    print('making sparse dictionary dense')
    for key in keys:
        if 'bias' in key:
            continue
        data = d[key]
        dense_struct = to_dense(data)
        d[key] = dense_struct
    return d
