import torch
import torch.nn as nn
import torch.autograd as grad
import torch
import math

import numpy as np
import numpy as np
from sklearn.cluster import KMeans
import scipy.signal as sgnt
from scipy.sparse import csc_matrix, csr_matrix


'''
some order regarding the dimension in each article and in this implemantation:
Wolf :
Signal comes in with 1xT (row, T elements in a row)
Than it goes to encoder (1x1 conv) that transform it to 
NxT' , where T' = 2T/L - 1, L = compression factor.
Than the output is chunked into R chunks - R = ceil(2T'/k) + 1
where each chunk is length k, hop size P = k/2
now the tensor is BxNxKxR  - B - batch_size,N_channel,K_chunk_length,R_numChunks.
next the tensor goes 




Tensor
Batch_SizexChannel_SizexWidthxLengthxSpeakers?
'''
def amin(x):
    mini = np.min(x)
    x = x.tolist()
    res = x.index(mini.item())
    return res
def clear_sound(sig,bw = 50, fs = 8000, fc = [0,1000,2000,3000,4000]):
          f,t,Zxx = sgnt.stft(sig,fs)
          Zxx_null = Zxx
          #rsmp = sgnt.resample_poly
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
          #f_const = int(fs/(fs-bw*10) * 100)
          #sig_null = rsmp(rsmp(sig_null,100,f_const),f_const,100)
          return sig_null

def tupple_p(blocknum,dict_weights):
    p1_w = to_sparse_encoded_tupple(dict_weights['seperation_net.blocklist.{}.P1.0.weight'.format(blocknum)])
    p1_b = dict_weights['seperation_net.blocklist.{}.P1.0.bias'.format(blocknum)]
    p2_w = to_sparse_encoded_tupple(dict_weights['seperation_net.blocklist.{}.P2.0.weight'.format(blocknum)])
    p2_b = dict_weights['seperation_net.blocklist.{}.P2.0.bias'.format(blocknum)]
    return (p1_w,p1_b,p2_w,p2_b)


def tupple_w_rnn(blocknum,number,dict_weights):
    wih = to_sparse_encoded_tupple(dict_weights['seperation_net.blocklist.{}.rnn_{}.weight_ih_l0'.format(blocknum,number)])
    whh = to_sparse_encoded_tupple(dict_weights['seperation_net.blocklist.{}.rnn_{}.weight_hh_l0'.format(blocknum,number)])
    bih = dict_weights['seperation_net.blocklist.{}.rnn_{}.bias_ih_l0'.format(blocknum,number)]
    bhh = dict_weights['seperation_net.blocklist.{}.rnn_{}.bias_hh_l0'.format(blocknum,number)]
    wih_rev = to_sparse_encoded_tupple(dict_weights['seperation_net.blocklist.{}.rnn_{}.weight_ih_l0_reverse'.format(blocknum,number)])
    whh_rev = to_sparse_encoded_tupple(dict_weights['seperation_net.blocklist.{}.rnn_{}.weight_hh_l0_reverse'.format(blocknum,number)])
    bih_rev = dict_weights['seperation_net.blocklist.{}.rnn_{}.bias_ih_l0_reverse'.format(blocknum,number)]
    bhh_rev = dict_weights['seperation_net.blocklist.{}.rnn_{}.bias_hh_l0_reverse'.format(blocknum,number)]
    wb_ih = (wih,bih)
    wb_hh = (whh,bhh)
    wb_ih_rev = (wih_rev,bih_rev)
    wb_hh_rev = (whh_rev,bhh_rev)
    w = (wb_ih,wb_hh,wb_ih_rev,wb_hh_rev)
    return w


#rnn_1_w,rnn_2_w,rnn_3_w,rnn_4_w,P1_w,P1_b,P2_w,P2_b = w[i]
def get_weights_list(dict_weights):
    w = []
    for layer in range(3):
        rnn = []
        for rnn_num in range(1,5):
            print('starts: block {} rnn {}'.format(layer,rnn_num))
            rnn.append(tupple_w_rnn(layer,rnn_num,dict_weights))
        p1_w,p1_b,p2_w,p2_b = tupple_p(layer,dict_weights)
        w.append((rnn[0],rnn[1],rnn[2],rnn[3],p1_w,p1_b,p2_w,p2_b))
    return w


def si_snr(x,y):
    x = x.squeeze()
    y = y.squeeze()
    eps = 1e-8
    si_tild = torch.sum(x*y+eps)*x/torch.sum(eps+x*x)
    ei_tild = y-si_tild
    si_snr = 10*torch.log10(torch.sum(si_tild*si_tild+eps)/torch.sum(eps+ei_tild*ei_tild))
    return si_snr
def convert_list(lst):
    return (*lst, )
def upit_loss(s,shat):
    '''
    s - BxTxC
    shat BlockxCxBx1xT
    '''
    permute_vec = [[0,1],[1,0]]
    num_blocks = shat.shape[0]
    tot_loss = 0
    batch_size = s.shape[0]
    res_block = []
    for block in range(num_blocks):
        s_hat_block = shat[block,:,:,:,:].squeeze()
        si_block = torch.tensor([-np.inf]).cuda()
        res_batch = []
        for b in range(batch_size):
            s_true_batch = s[b,:,:].squeeze()
            s_est_batch = s_hat_block[:,b,:].squeeze()
            res_perm = []
            for perm in permute_vec:
                s0 = s_est_batch[perm[0],:].squeeze()
                s1 = s_est_batch[perm[1],:].squeeze()
                s_true_0 = s_true_batch[:,0].squeeze()
                s_true_1 = s_true_batch[:,1].squeeze()
                res_perm.append((si_snr(s_true_0,s0)+si_snr(s_true_1,s1))*.5)
            res_batch.append(-torch.max(torch.stack(res_perm)))
        res_block.append(torch.mean(torch.stack(res_batch)))
    total_loss = torch.mean(torch.stack(res_block))/2
    return total_loss




#64,16,256,3,int(T*8e3),multi_loss = False, hidden_size = 128

class WolfModel(nn.Module):
    def __init__(self,N,L,K,num_blocks,T,multi_loss = False, hidden_size = 64 ,bidirectional = True,MulCat = False,weights_for_seperation = []):
        super(WolfModel, self).__init__()
        self.N,self.L,self.K,self.num_blocks= N,L,K,num_blocks
        simple_net = nn.Conv1d(in_channels=1,out_channels=N,kernel_size=L,stride=L//2)
        xx,ss = self._Segmentation(simple_net(torch.randn(1,1,T)),self.K)
        self.R = xx.shape[3]
        R = self.R
        print('N = {} L = {} K = {} R = {}'.format(N,L,K,R))
        self.ReLU = nn.ReLU()
        self.multi_loss = multi_loss
        self.MulCat = MulCat
        self.bidirectional = bidirectional
        #print(xx.shape)
        self.num_speakers = 2
        self.eps = 1e-6
        self.activation = nn.ReLU()
        #print(R,self.num_speakers*R,1)
        self.encoding = nn.Conv1d(in_channels=1,out_channels=N,kernel_size=L,stride=L//2,bias = False)
        self.bn = nn.BatchNorm1d(N,eps = self.eps)
        #self.encoding = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=N,kernel_size=L,stride=L//2,bias = False),nn.BatchNorm1d(N),nn.ReLU())
        self.seperation_net = SeperationNet(in_channels= N,out_channels = N,hidden_channels = hidden_size,num_layer = num_blocks,num_speakers = 2,bidirectional = self.bidirectional,MulCat = self.MulCat,w=weights_for_seperation)
        self.decoding = nn.Sequential(nn.PReLU(),nn.Conv2d(in_channels=R,out_channels=self.num_speakers*R,kernel_size=1,bias=False),nn.BatchNorm2d(self.num_speakers*R,eps = self.eps),nn.ReLU())
        self.deconv = nn.Sequential(nn.ConvTranspose1d(in_channels=N,out_channels=1,kernel_size=L,stride=L//2,bias = False))
        # According to the paper we should use weights initialization
        # Reading Xavier weight initialization paper for symmetric non linearty (if we used relu we use Kaimen)
        # we normalize by using the factor sqrt(6)/sqrt(num_input + num_output)
        #for parameter in self.parameters():
        #    nn.init.xavier_uniform_(parameter)

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, R , C]
           gap: padding length
           output: [B, N, L ,C] 
        '''
        B, N, K, R, C = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2,C)
        input1 = input[:, :, :, :K,:].contiguous().view(B, N, -1,C)[:, :, P:,:]
        input2 = input[:, :, :, K:,:].contiguous().view(B, N, -1,C)[:, :, :-P,:]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap,:]
        return input


    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap
    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, R]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)
        return input.contiguous(), gap
    
    def forward(self, input):
        y = self.ReLU(self.bn(self.encoding(input)))
        #print('shape before')
        #print(y.shape)
        Chunks,gap = self._Segmentation(y,self.K)
        B,N,K,R = Chunks.shape
        #print('shape after')
        #print(Chunks.shape)
        new_outputs = [self.seperation_net.blocklist[0](Chunks)]
        for i in range(1,self.num_blocks):
            if self.multi_loss:
                new_outputs.append(self.seperation_net.blocklist[i](new_outputs[i-1])) # BNKR
            else :
                new_outputs =[self.seperation_net.blocklist[i](new_outputs[0])] # BNKR
        new_outputs = [self._over_add(self.decoding(x.permute(0,3,2,1)).permute(0,3,2,1).contiguous().view(B,N,K,R,self.num_speakers),gap) for x in new_outputs]
        deconv_outputs = []
        for x in new_outputs:
            tmp = []
            for ix in range(self.num_speakers):
                s_tmp = self.deconv(x[:,:,:,ix])
                t_tmp = torch.zeros(s_tmp.shape).to('cpu')
                for b in range(s_tmp.shape[0]):
                    t_tmp[b,:,:]= s_tmp[b,:,:]
                tmp.append(t_tmp)
            deconv_outputs.append(torch.stack(tmp))
        return torch.stack(deconv_outputs)

        
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                    continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


def apply_weight_sharing(mat, bits=12):
    """
    Applies weight sharing to the given model
    """
    #for p in model.parameters():
    #        if 'weight' not in name:
    #            continue
    return mat
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






class SeperationNet(nn.Module):
    def __init__(self,in_channels,out_channels,hidden_channels,num_layer,num_speakers,bidirectional,MulCat,w):
        super(SeperationNet,self).__init__()
        self.blocklist = nn.ModuleList([])
        for i in range(num_layer):
            rnn_1_w,rnn_2_w,rnn_3_w,rnn_4_w,P1_w,P1_b,P2_w,P2_b = w[i]
            self.blocklist.append(SepBlock(out_channels, hidden_channels,num_speakers,bidirectional,MulCat,rnn_1_w,rnn_2_w,rnn_3_w,rnn_4_w,P1_w,P1_b,P2_w,P2_b))

class SepBlock(nn.Module):
        def __init__(self,out_channels,hidden_channels,num_speakers,bidirectional,MulCat,rnn_1_w,rnn_2_w,rnn_3_w,rnn_4_w,P1_w,P1_b,P2_w,P2_b):
            super(SepBlock,self).__init__()
            self.bidirectional = bidirectional
            self.MulCat = MulCat
            self.rnn_1 = LSTM_BI_DIR(rnn_1_w,hidden_channels,out_channels)
            if self.MulCat:
                self.rnn_2 = LSTM_BI_DIR(rnn_2_w,hidden_channels,out_channels)
                self.rnn_4 = LSTM_BI_DIR(rnn_4_w,hidden_channels,out_channels)
            self.rnn_3 = LSTM_BI_DIR(rnn_3_w,hidden_channels,out_channels)
            self.gn1 = nn.GroupNorm(num_groups=1,num_channels=out_channels,eps=1e-8,affine=True)
            self.gn2 = nn.GroupNorm(num_groups=1,num_channels=out_channels,eps=1e-8,affine=True)
            self.P1 = nn.Sequential(Sparse_Linear(to_dense(P1_w),P1_b))
            self.P2 = nn.Sequential(Sparse_Linear(to_dense(P2_w),P2_b))
        def forward(self,input):
            B,N,K,R = input.shape
            # make tensor act on short term dim
            short_term = input.permute(0,3,2,1).contiguous().view(B*R, K, N)
            #print('shape after agg')
            #print(short_term.shape)
            short_term_fin  = self.rnn_1(short_term)
            if self.MulCat:
                short_term_fin1  = self.rnn_2(short_term)
            else:
                short_term_fin1 = 1
            short_term_con = torch.cat((short_term_fin*short_term_fin1,short_term), dim = 2)
            transition_tensor = self.gn1(self.P1(short_term_con.contiguous().view(B*R*K, -1)).view(B*R, K, N).view(B,R,K,N).permute(0, 3, 2, 1).contiguous()) + input
            modified_transition = transition_tensor.permute(0, 2, 3, 1).contiguous().view(B*K, R, N)
            long_term1 = self.rnn_3(modified_transition)
            if self.MulCat:
                long_term2 = self.rnn_4(modified_transition)
            else:
                long_term2 = 1
            long_term_con = torch.cat((long_term1*long_term2,modified_transition), dim = 2)
            output = self.gn2(self.P2(long_term_con.contiguous().view(B*K*R, -1)).view(B*K, R, N).view(B,K,R,N).permute(0, 3, 1, 2).contiguous()) + transition_tensor
            return output

class Encoder_Decoder(nn.Module):
    def __init__(self,N,L,K,num_blocks,T):
        super(Encoder_Decoder, self).__init__()
        self.N,self.L,self.K,self.num_blocks= N,L,K,num_blocks
        self.eps = 1e-5
        self.relu = nn.ReLU()
        self.encoding = nn.Conv1d(in_channels=1,out_channels=N,kernel_size=L,stride=L//2,bias = False)
        self.deconv = nn.Sequential(nn.ConvTranspose1d(in_channels=N,out_channels=1,kernel_size=L,stride=L//2,bias = False))
    def forward(self,input):
      out = self.deconv(self.relu(self.encoding(input)))
      return out



import math
import torch.nn as nn
import torch
import numpy as np

class LSTM_BI_DIR( nn.Module):
    def __init__(self,w,hidden_size,out_channels):
        super (LSTM_BI_DIR, self).__init__()
        wb_ih,wb_hh,wb_ih_rev,wb_hh_rev = w
        self.wih,self.bih = wb_ih
        self.whh,self.bhh = wb_hh
        self.wih_rev,self.bih_rev = wb_ih_rev
        self.whh_rev,self.bhh_rev = wb_hh_rev
        self.lstm_forward = LSTM_Sparse(hidden_size,self.wih,self.bih,self.whh,self.bhh,out_channels,self.wih_rev,self.bih_rev,self.whh_rev,self.bhh_rev)
        #self.lstm_rev = LSTM_Sparse(hidden_size,self.wih_rev,self.bih_rev,self.whh_rev,self.bhh_rev,out_channels,self.wih_rev,self.bih_rev,self.whh_rev,self.bhh_rev)
    def forward(self,input):
        h1 = self.lstm_forward(input)
        #h2 = self.lstm_rev(self.invert(input))
        return h1 #torch.cat((h1,h2),dim = 2)

    def invert(self,input):
        return input.flip(0)


class LSTM_Sparse(nn.Module):

    def __init__(self, hidden_size, weight_ih,bias_ih, weight_hh,bias_hh,out_channels,weight_ih_r,bias_ih_r, weight_hh_r,bias_hh_r):
        super(LSTM_Sparse, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=out_channels,hidden_size=hidden_size,num_layers=1,batch_first=True,bidirectional=True)
        self.lstm.weight_ih_l0.data =  to_dense(weight_ih)
        self.lstm.weight_hh_l0.data =  to_dense(weight_hh)
        self.lstm.bias_ih_l0.data =  bias_ih
        self.lstm.weight_hh_l0_reverse.data =  to_dense(weight_hh_r)
        self.lstm.weight_ih_l0_reverse.data =  to_dense(weight_ih_r)
        self.lstm.bias_ih_l0_reverse.data =  bias_ih_r
        self.lstm.bias_hh_l0_reverse.data =  bias_hh_r
        '''
        self.i2h = Sparse_Linear(to_dense(weight_ih).to_sparse(),bias_ih)
        self.h2h = Sparse_Linear(to_dense(weight_hh).to_sparse(),bias_hh)
        self.h = None #torch.zeros(256,68,hidden_size)
        self.c = None #torch.zeros(68, hidden_size)
        #self.reset_parameters()
        '''
    #def sample_mask(self):
    #    keep = 1.0 - self.dropout
    #    self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x):
        '''
        # do_dropout = self.training and self.dropout > 0.0
        batch_size,seq = x.shape[0],x.shape[1]
        if self.h == None:
            self.h = torch.zeros(seq,batch_size,self.hidden_size)
            self.c = torch.zeros(batch_size,self.hidden_size)
        batch_size,seq = x.shape[0],x.shape[1]
        #for i in range(batch_size):
        for j in range(seq):
            h = self.h[max(0,j-1),:,:].squeeze()
            preact = self.i2h(x[:,j,:].squeeze()) + self.h2h(h)
            ingate, forgetgate, cellgate, outgate = preact.chunk(4, 1)
            ingate = ingate.sigmoid()
            forgetgate = forgetgate.sigmoid()
            cellgate = cellgate.tanh()
            outgate = outgate.sigmoid()

            cy = (forgetgate * self.c) + (ingate * cellgate)
            hy = outgate * cy.tanh()

            self.h[j,:,:] = hy
            self.c = cy


        
        for b in range(batch_size):
            xx = x[:,b,:].squeeze()
            # Linear mappings
            preact = self.i2h(xx.permute(1,0)) + self.h2h(h[b,:,:].squeeze().permute(1,0))
            # activations
            gates = preact[:, :3 * self.hidden_size].sigmoid()
            g_t = preact[:, 3 * self.hidden_size:].tanh()
            i_t = gates[:, :self.hidden_size]
            f_t = gates[:, self.hidden_size:2 * self.hidden_size]
            o_t = gates[:, -self.hidden_size:]
            c_t = torch.mul(self.c[b,:,:].squeeze(), f_t) + torch.mul(i_t, g_t)
            h_t = torch.mul(o_t, c_t.tanh())
            h_t = h_t.view(1, h_t.size(0), -1)
            c_t = c_t.view(1, c_t.size(0), -1)
            self.h[b,:,:] = h_t.unsqueeze(0)
            self.c[b,:,:] = c_t.unsqueeze(0)
        '''
        h,c = self.lstm(x)
        return h #self.h.permute(1,0,2)#, (h_t, c_t)


'''
def to_sparse_encoded_tupple(mat):
    rows,cols = mat.shape
    values = torch.unique(mat).tolist()
    pos = []
    indecies = []
    print('values length is {}'.format(len(values)))
    cnt = 0
    total = rows*cols
    for i in range(rows):
        for j in range(cols):
            if mat[i,j] != 0:
                if cnt % 100 == 0:
                    print('@ iter {}, total array num {}'.format(cnt,total))
                val = mat[i,j]
                indecies.append(values.index(val))
                pos.append(sub2ind([i,j],mat.shape))
                cnt +=1
            else:
                continue
    return torch.tensor(pos,dtype=torch.int16),torch.tensor(indecies,dtype=torch.uint8),values,torch.tensor(mat.shape,dtype=torch.int16)
'''

def prepare_mapping(values):
    dicti = {}
    for i,v in enumerate(values):
        dicti['{}'.format(v)] = i
    return dicti

def to_dense(mat_dict):
    return mat_dict
    new_mat = torch.zeros(mat_dict['orig_shape'][0]*mat_dict['orig_shape'][1])
    for cnt,i in enumerate(mat_dict['ind']):
        new_mat[i] = mat_dict['values'][mat_dict['labels_'][cnt]]
    new_mat = new_mat.view(mat_dict['orig_shape'])
    return new_mat


def to_sparse_encoded_tupple(mat):
    return mat
    orig_shape = mat.shape
    mat_n = mat#apply_weight_sharing(mat, bits=12)
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


def to_mat(position,indecies,values,shape):
    mat = torch.zeros(shape)
    for cnt,p in enumerate(position):
        i,j = ind2sub(p,shape)
        mat[i,j] = values[indecies[cnt]]
    return mat.to_sparse()

def ind2sub(position,shape):
    return np.unravel_index(position,shape)

def sub2ind(position_vec,shape):
    return np.ravel_multi_index(position_vec,shape)






class Sparse_Linear(nn.Module):
    def __init__(self,weight, bias):
        super(Sparse_Linear, self).__init__()
        #position_w,indecies_w,values_w,shape_w = weight
        #position_b,indecies_b,values_b,shape_b = bias
        #weight = to_mat(position_w,indecies_w,values_w,shape_w)
        #bias = to_mat(position_b,indecies_b,values_b,shape_b)
        self.ln = nn.Linear(320,64)
        self.w = weight.to('cpu')
        print(self.w)
        self.bias = bias.to('cpu')
        self.ln.bias.data = self.bias
        self.ln.weight.data = self.w
    def forward(self,input):
        '''
        output = torch.sparse.mm(self.w, input)
        bias = self.bias.unsqueeze(1).repeat(1,output.shape[1])
        output += bias
        
        output = torch.sparse.mm(self.w,input.permute(1,0)).permute(1,0)
        for b in range(output.shape[0]):
            output[b,:] += self.bias
        
        #output = torch.sparse.addmm(self.bias, self.w.permute(1,0),input, beta=1, alpha=1)
        #return output
        '''
        return self.ln(input)

    

def _matmul(mat,mat1,mat2):
    if mat1.dim() == 2:
        return torch.sparse.addmm(mat, mat1,mat2, beta=1, alpha=1)
    


