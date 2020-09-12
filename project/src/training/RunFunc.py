import torch
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
import numpy as np
import fully_working_model as mdl
import dataloader
from random import shuffle
import os
import time
from fully_working_model import Encoder_Decoder
import matplotlib.pyplot as plt
import cProfile, pstats, io
from pstats import SortKey
from torch.autograd import Variable
import IPython
import warnings 
import torch
import numpy as np
from sklearn.cluster import KMeans
import scipy.signal as sgnt
import librosa
from scipy.sparse import csc_matrix, csr_matrix
warnings.filterwarnings("ignore")

'''
pr = cProfile.Profile()
pr.enable()
# ... do something ...
'''
def check_sparisity(md):
    whole = 0
    nonz = 0
    with torch.no_grad():
        for p in md.parameters():
            x = p.data.detach()
            whole += x.view(-1).numel()
            nonz += x.view(-1).nonzero().numel()
    return nonz/whole,whole
def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

def measure_epoch_time(t=0):
    return time.time() - t


def load_to_ram(dl):
    dl_train = []
    for p in dl:
        dl_train.append(torch.load(p))
    return dl_train
def amin(x):
    mini = np.min(x)
    x = x.tolist()
    res = x.index(mini.item())
    return res
def clear_sound(sig,bw = 50, fs = 8000, fc = [0,1000,2000,3000,4000]):
          f,t,Zxx = sgnt.stft(sig,fs)
          Zxx_null = Zxx
          rsmp = librosa.resample
          for ix,fc_c in enumerate(fc):
              if (ix == len(fc)-1) |(ix == 0):
                  bw_c = bw * 5
              else:
                  bw_c = bw
              if ix !=0:
                  min_ix = amin(abs(f-(fc_c -bw_c)))
              else:
                  min_ix = 0
              if ix == len(fc) - 1:
                  max_ix = len(f)-1
              else:
                  max_ix = amin(abs(f - (fc_c + bw_c)))
              Zxx_null[min_ix:max_ix,:] *= 1e-6
          _,sig_null = sgnt.istft(Zxx_null,fs)
          sig_null = rsmp(rsmp(sig_null,fs,fs-bw*10),fs-bw*10,fs)
          return sig_null

def apply_weight_sharing(model, bits=5):
          """
          Applies weight sharing to the given model
          """
          for p in model.parameters():
      #        if 'weight' not in name:
      #            continue
              data = p.data
              if data.numel() < 2**bits:
                  continue
              weight = data.cpu().numpy()
              shape = weight.shape
      #        print(shape)
              mat = weight.reshape(-1,1)
              mat = csc_matrix(mat)
              min_ = min(mat.data)
              max_ = max(mat.data)
              space = np.linspace(min_, max_, num=2**bits)
              kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
              kmeans.fit(mat.data.reshape(-1,1))
              new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
              mat.data = new_weight
              mat_n = mat.toarray()
              p.data = torch.from_numpy(mat_n.reshape(shape)).to('cuda')


def load_dl(dl_train_type,inp1,inp2):
    if dl_train_type == 'files':
        x1,y1 = torch.load(inp1)
        x2,y2 = torch.load(inp2)
    else :
        x1,y1 = inp1
        x2,y2 = inp2
    return (x1,y1) , (x2,y2)

class improvement_graph():
    def __init__(self):
        self.fig,self.ax = plt.subplots()
        self.count = 0
        self.arr = []
    def add_point(self,y):
        self.arr.append(y.cpu().numpy())
        self.ax.plot(self.arr)
        self.ax.grid()
        self.ax.set_title('SI-SNR Improvement (last layer) epoch {}'.format(self.count))
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('SI-SNR Improvement [dB]')
        self.count += 1
        self.fig.show()


si_snr = mdl.si_snr
def loss_by_relative_si_snr(xx,shat,yy):
    n_block = shat.shape[0]
    loss = 0
    for block in range(shat.shape[0]):
        shat_tmp = shat[block,:,:,:,:].squeeze()
        perm_vec = [[1,0],[0,1]]
        res_batch = []
        for b in range(batch_size):
            s_true_batch = xx[b,:,:].squeeze()
            s_est_batch = shat_tmp[:,b,:].squeeze()
            tmptest = (s_est_batch.detach() > 1e-3).any()
            if not tmptest:
                continue
            y_b = yy[b,:,:].squeeze()
            res_perm = []
            for perm in perm_vec:
                s0 = s_est_batch[perm[0],:].squeeze()
                s1 = s_est_batch[perm[1],:].squeeze()
                s_true_0 = s_true_batch[:,0].squeeze()
                s_true_1 = s_true_batch[:,1].squeeze()
                res_perm.append((si_snr(s_true_0,s0)+si_snr(s_true_1,s1))-(si_snr(s_true_0,y_b)+si_snr(s_true_1,y_b)))
            res_batch.append(torch.max(torch.stack(res_perm)))
        loss += -torch.mean(torch.stack(res_batch))/2/n_block
    return loss

def si_snr_i(xx,shat,yy):
    with torch.no_grad():
        shat = shat[-1,:,:,:,:].squeeze()
        perm_vec = [[1,0],[0,1]]
        res_batch = []
        for b in range(batch_size):
            s_true_batch = xx[b,:,:].squeeze()
            s_est_batch = shat[:,b,:].squeeze()
            y_b = yy[b,:,:].squeeze()
            res_perm = []
            for perm in perm_vec:
                s0 = s_est_batch[perm[0],:].squeeze()
                s1 = s_est_batch[perm[1],:].squeeze()
                s_true_0 = s_true_batch[:,0].squeeze()
                s_true_1 = s_true_batch[:,1].squeeze()
                res_perm.append((si_snr(s_true_0,s0)+si_snr(s_true_1,s1))*.5-(si_snr(s_true_0,y_b)+si_snr(s_true_1,y_b))*.5)
            res_batch.append(torch.max(torch.stack(res_perm)))
        return -torch.mean(torch.stack(res_batch))

def gen_mask(p,q):
        shape = p.shape
        p = p.cpu().numpy().reshape(-1)
        val = np.quantile(np.abs(p),q)
        p =np.abs(p) > val
        return to_var(torch.tensor(p.reshape(shape)).float(), requires_grad=False)

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def apply_pruning(masks,model):
    cnt = 0
    for p in model.seperation_net.parameters():
        if len(p.data.size()) != 1:
            shape = p.data.shape
            p.data = (p.data.flatten()*masks[cnt]).view(shape)
            cnt += 1
    return model



batch_size = 2 #num of mixtures in batch
target_sr = 8e3 #samplerate to process in [sps]
dl_train_type = 'files' # files
if dl_train_type == 'files':
    T = 4
    data_dir = r'C:\Users\shaha\Documents\DeepGit\DeepLearningSeminar\project\data'
    data_file_list = os.listdir(data_dir)
    dl_train_orig = [data_dir + '/' + x for x in data_file_list if '.pt' in x]
    shuffle(dl_train_orig) # make sure we get all mixtures of ppl and segments
    tot_ds_len = len(dl_train_orig)
    dl_train_precentile = 1
    dl_test_precentile = .2
    fin_ix = int(tot_ds_len*dl_train_precentile)
    dl_train = dl_train_orig[:fin_ix]
    dl_test = dl_train_orig[fin_ix+1:]
    dl_train = load_to_ram(dl_train)
    dl_train_type = 'agg'
else:
    T = 3.8
    dl_train = torch.load('data_short.pt')

pruning_stage = 1
pruning_factor = .87

if pruning_stage == 0 :
    model = mdl.WolfModel(64,16,256,3,int(T*8e3),multi_loss = False, hidden_size = 128 ,bidirectional = False,MulCat = True)
    config = '64 16 256 3 int(T*8e3) multi_loss True hidden_size 64 bidirectional False MulCat True'
    encoder_decoder = torch.load('encodr_decoder_weights.pth')

    model.encoding.load_state_dict(encoder_decoder.encoding.state_dict()) 
    model.deconv.load_state_dict(encoder_decoder.deconv.state_dict()) 
    model.bn.load_state_dict(encoder_decoder.bn.state_dict()) 
    for e,d in zip(model.encoding.parameters(),model.deconv.parameters()):
        e.requires_grad = False
        d.requires_grad = False
    print(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor= .90,patience=3,verbose=True,threshold=.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,2,.90,-1)
    lr = 5e-4
    num_epochs = 130
    clip_norm = 5
    step_size = 2
    decay_factor = .98
    checkpoint = 15
    first_time_negative = False
    n_train = len(dl_train)//2
    batch_size_summation = 1
    for epoc in range(0,num_epochs):
        t_start = time.time()
        grad_accum = 0
        cum_error = 0
        idx = 0
        si_snr_rel = 0
        si_snr_i_tmp = 0
        optimizer.zero_grad()
        #shuffle(dl_train) # make sure that the batch average gradient is not affected by the same examples each epoch
        for inp1,inp2 in pairwise(dl_train): #,inp3,inp4,inp5,inp6
            (x1,y1) , (x2,y2)= load_dl(dl_train_type,inp1,inp2)#,inp3,inp4,inp5,inp6) #, (x3,y3), (x4,y4), (x5,y5), (x6,y6) 
            x = torch.stack((x1,x2),0) #,x3,x4,x5,x6
            y = torch.stack((y1,y2),0) #,y3,y4,y5,y6
            grad_accum += 1
            #optimizer.zero_grad()
            yy = y.to(device)
            xx = x.to(device)
            shat = model(yy.view(batch_size,1,-1))
            loss = loss_by_relative_si_snr(xx,shat,yy)
            cum_error += loss.item()
            error = loss/batch_size_summation
            error.backward()
            if grad_accum == batch_size_summation:
                torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                grad_accum = 0
            #si_snr_i_tmp += si_snr_i(xx,shat,yy)
            idx += 1
            precentile_done = round(100*(idx + 1)/n_train)
            progress_symbols = int(np.floor(precentile_done*80/100))
            print('\r['
                    + ('#')*progress_symbols
                    + (' ')*(80 - progress_symbols)
                    + ']' +
                    ' Epoch {}/{} progress {}/100%'.format(epoc + 1, num_epochs, precentile_done), end='')
        t_epoch = measure_epoch_time(t_start)
        print('\n')
        print('*'*33 + 'epoch  results' + '*'*33)
        print('epoch {} , si_snr {:+.2f},si snr improvement {:+.2f}, epoch time {:+.2f} [min], estimated time last {:+.2f} [min]'.format(epoc,-cum_error/idx,-si_snr_i_tmp/idx,t_epoch/60,(num_epochs- epoc - 1)*(t_epoch/60)))
        print('*'*80)
        scheduler.step()
        #if epoc%step_size == 0:
        #    optimizer.param_groups[0]['lr'] = lr*decay_factor
        #    lr = lr*decay_factor
        if -cum_error/idx > checkpoint:
            print('save model')
            torch.save(model,config + '{}.pth'.format(epoc))
else :
    #model = mdl.WolfModel(64,16,256,3,int(T*8e3),multi_loss = False, hidden_size = 128 ,bidirectional = False,MulCat = True)
    model = torch.load('op_model_12_99.pth')
    #model.load_state_dict(ref_model.state_dict()) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3*(.9**13))
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor= .90,patience=3,verbose=True,threshold=.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,2,.90,-1)
    lr = 5e-4
    num_epochs = 20
    clip_norm = 5
    step_size = 2
    decay_factor = .98
    checkpoint = 15
    first_time_negative = False
    n_train = len(dl_train)//2
    batch_size_summation = 1
    pruning_factor = .2
    score_thd = 9.5
    fine_tune_iter = 2
    cnt_prune = 0
    masks = []
    for p in model.seperation_net.parameters():
        if len(p.data.size()) != 1:
            masks.append(gen_mask(p.data.flatten(),pruning_factor))
    for e,d in zip(model.encoding.parameters(),model.deconv.parameters()):
        e.requires_grad = False
        d.requires_grad = False

    for epoc in range(0,num_epochs):
        if (cnt_prune > fine_tune_iter):
            if (-cum_error/idx) > score_thd: 
                if pruning_factor < 1:#.82:
                    cnt_prune = 0
                    #pruning_factor = pruning_factor/.7
                    masks = []
                    for p in model.seperation_net.parameters():
                        if len(p.data.size()) != 1:
                            masks.append(gen_mask(p.data.flatten(),pruning_factor))
        cnt_prune += 1
        t_start = time.time()
        grad_accum = 0
        cum_error = 0
        idx = 0
        si_snr_rel = 0
        si_snr_i_tmp = 0
        optimizer.zero_grad()
        model = apply_pruning(masks,model)
        #shuffle(dl_train) # make sure that the batch average gradient is not affected by the same examples each epoch
        for inp1,inp2 in pairwise(dl_train): #,inp3,inp4,inp5,inp6
            (x1,y1) , (x2,y2)= load_dl(dl_train_type,inp1,inp2)#,inp3,inp4,inp5,inp6) #, (x3,y3), (x4,y4), (x5,y5), (x6,y6) 
            x = torch.stack((x1,x2),0) #,x3,x4,x5,x6
            y = torch.stack((y1,y2),0) #,y3,y4,y5,y6
            grad_accum += 1
            #optimizer.zero_grad()
            yy = y.to(device)
            xx = x.to(device)
            #model = apply_pruning(masks,model)
            #cnt = 0
            #for p in model.seperation_net.parameters():
            #    if len(p.data.size()) != 1:
            #        shape = p.data.shape
            #        p.data = (p.data.flatten()*masks[cnt]).view(shape)
            #        cnt += 1

            shat = model(yy.view(batch_size,1,-1))
            loss = loss_by_relative_si_snr(xx,shat,yy)
            cum_error += loss.item()
            error = loss/batch_size_summation
            error.backward()
            if grad_accum == batch_size_summation:
                torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                grad_accum = 0
            #si_snr_i_tmp += si_snr_i(xx,shat,yy)
            idx += 1
            '''
            cnt = 0
            for p in model.seperation_net.parameters():
                if len(p.data.size()) != 1:
                    shape = p.data.shape
                    p.data = (p.data.flatten()*masks[cnt]).view(shape)
                    cnt += 1
            '''
            if idx>1000:
                break
            precentile_done = round(100*(idx + 1)/1000)
            progress_symbols = int(np.floor(precentile_done*80/100))
            print('\r['
                    + ('#')*progress_symbols
                    + (' ')*(80 - progress_symbols)
                    + ']' +
                    ' Epoch {}/{} progress {}/100%'.format(epoc + 1, num_epochs, precentile_done), end='')
        t_epoch = measure_epoch_time(t_start)
        print('\n')
        print('*'*33 + 'epoch  results' + '*'*33)
        print('epoch {} , si_snr {:+.2f},sparisity {:+.2f}, epoch time {:+.2f} [min], estimated time last {:+.2f} [min]'.format(epoc,-cum_error/idx,check_sparisity(model),t_epoch/60,(num_epochs- epoc - 1)*(t_epoch/60)))
        print('*'*80)
        scheduler.step()
        #if epoc%step_size == 0:
        #    optimizer.param_groups[0]['lr'] = lr*decay_factor
        #    lr = lr*decay_factor
        if epoc % 5 == 0:
            print('save model')
            torch.save(model,'prune_{}.pth'.format(epoc))
