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
warnings.filterwarnings("ignore")


import torch
import torch.nn as nn
import torch.autograd as grad
import torch
import math

import numpy as np


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






class WolfModel(nn.Module):
    def __init__(self,N,L,K,num_blocks,T,multi_loss = False, hidden_size = 64 ,bidirectional = True,MulCat = False):
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
        self.seperation_net = SeperationNet(in_channels= N,out_channels = N,hidden_channels = hidden_size,num_layer = num_blocks,num_speakers = 2,bidirectional = self.bidirectional,MulCat = self.MulCat)
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
                t_tmp = torch.zeros(s_tmp.shape).to('cuda')
                for b in range(s_tmp.shape[0]):
                    t_tmp[b,:,:]= s_tmp[b,:,:]
                tmp.append(t_tmp)
            deconv_outputs.append(torch.stack(tmp))
        return torch.stack(deconv_outputs)




class SeperationNet(nn.Module):
    def __init__(self,in_channels,out_channels,hidden_channels,num_layer,num_speakers,bidirectional,MulCat):
        super(SeperationNet,self).__init__()
        self.blocklist = nn.ModuleList([])
        for i in range(num_layer):
            self.blocklist.append(SepBlock(out_channels, hidden_channels,num_speakers,bidirectional,MulCat))

class SepBlock(nn.Module):
        def __init__(self,out_channels,hidden_channels,num_speakers,bidirectional,MulCat):
            super(SepBlock,self).__init__()
            self.bidirectional = bidirectional
            self.MulCat = MulCat
            self.rnn_1 = nn.GRU(input_size=out_channels,hidden_size=hidden_channels,num_layers=1,batch_first=True,bidirectional=self.bidirectional)
            if self.MulCat:
                self.rnn_2 = nn.GRU(input_size=out_channels,hidden_size=hidden_channels,num_layers=1,batch_first=True,bidirectional=self.bidirectional)
                self.rnn_4 = nn.GRU(input_size=out_channels,hidden_size=hidden_channels,num_layers=1,batch_first=True,bidirectional=self.bidirectional)
            self.rnn_3 = nn.GRU(input_size=out_channels,hidden_size=hidden_channels,num_layers=1,batch_first=True,bidirectional=self.bidirectional)
            self.gn1 = nn.GroupNorm(num_groups=1,num_channels=out_channels,eps=1e-8,affine=True)
            self.gn2 = nn.GroupNorm(num_groups=1,num_channels=out_channels,eps=1e-8,affine=True)
            self.P1 = nn.Sequential(nn.Linear(hidden_channels*(2 if bidirectional else 1) + out_channels,out_channels))
            self.P2 = nn.Sequential(nn.Linear(hidden_channels*(2 if bidirectional else 1) + out_channels,out_channels))
        def forward(self,input):
            B,N,K,R = input.shape
            # make tensor act on short term dim
            short_term = input.permute(0,3,2,1).contiguous().view(B*R, K, N)
            short_term_fin,_  = self.rnn_1(short_term)
            if self.MulCat:
                short_term_fin1,_1  = self.rnn_2(short_term)
            else:
                short_term_fin1 = 1
            short_term_con = torch.cat((short_term_fin*short_term_fin1,short_term), dim = 2)
            transition_tensor = self.gn1(self.P1(short_term_con.contiguous().view(B*R*K, -1)).view(B*R, K, N).view(B,R,K,N).permute(0, 3, 2, 1).contiguous()) + input
            modified_transition = transition_tensor.permute(0, 2, 3, 1).contiguous().view(B*K, R, N)
            long_term1,_s3 = self.rnn_3(modified_transition)
            if self.MulCat:
                long_term2,_s4 = self.rnn_4(modified_transition)
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
        self.encoding = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=N,kernel_size=L,stride=L//2,bias = False),nn.BatchNorm1d(N,eps = self.eps),nn.ReLU())
        self.deconv = nn.Sequential(nn.ConvTranspose1d(in_channels=N,out_channels=1,kernel_size=L,stride=L//2,bias = False))
    def forward(self,input):
      out = self.deconv(self.encoding(input))
      return out
import numpy as np
from random import shuffle
import os
import time
import warnings 
warnings.filterwarnings("ignore")
from torch.autograd import Variable
'''
pr = cProfile.Profile()
pr.enable()
# ... do something ...
'''

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

def measure_epoch_time(t=0):
    return time.time() - t


def load_to_ram(dl):
    dl_train = []
    n = len(dl)
    for ix,p in enumerate(dl):
        dl_train.append(torch.load(p))
        print('progress in loading {}'.format(100*ix/n))
    return dl_train



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


si_snr = si_snr
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

def si_snr_i(xx,shat):
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
            res_perm = []
            for perm in perm_vec:
                s0 = s_est_batch[perm[0],:].squeeze()
                s1 = s_est_batch[perm[1],:].squeeze()
                s_true_0 = s_true_batch[:,0].squeeze()
                s_true_1 = s_true_batch[:,1].squeeze()
                res_perm.append((si_snr(s_true_0,s0)+si_snr(s_true_1,s1)))#-(si_snr(s_true_0,y_b)+si_snr(s_true_1,y_b)))
            res_batch.append(torch.max(torch.stack(res_perm)))
        loss += -torch.mean(torch.stack(res_batch))/n_block
    return loss
batch_size = 2 #num of mixtures in batch
target_sr = 8e3 #samplerate to process in [sps]
dl_train_type = 'agg' # files
if dl_train_type == 'files':
    T = 4
    data_dir = r'/content/drive/My Drive/Deep Learning Seminar/Project/DATA_AGG/data_long.pth'
    data_file_list = os.listdir(data_dir)
    dl_train_orig = [data_dir + '/' + x for x in data_file_list if '.pt' in x]
    shuffle(dl_train_orig) # make sure we get all mixtures of ppl and segments
    tot_ds_len = len(dl_train_orig)
    dl_train_precentile = .05
    dl_test_precentile = .2
    fin_ix = int(tot_ds_len*dl_train_precentile)
    dl_train = dl_train_orig[:fin_ix]
    dl_test = dl_train_orig[fin_ix+1:]
    dl_train = load_to_ram(dl_train)
    dl_train_type = 'agg'
else:
    T = 4
    dl_train = torch.load('data_long.pth')

teacher = torch.load(r"C:\Users\shaha\Documents\DeepGit\DeepLearningSeminar\project\trained_models\op_model_12_99.pth")
device = 'cuda'
teacher.to(device)
teacher.eval()
student =  WolfModel(32,16,256,4,int(4*8e3),multi_loss=False,hidden_size = 64,bidirectional = False,MulCat = True)
student.to(device)
'''
student.encoding.load_state_dict(teacher.encoding.state_dict()) 
student.deconv.load_state_dict(teacher.deconv.state_dict())
student.bn.load_state_dict(teacher.bn.state_dict()) 
for e,d in zip(student.encoding.parameters(),student.deconv.parameters()):
        e.requires_grad = False
        d.requires_grad = False
'''
student.train()
n_epochs = 20
n_train = len(dl_train)//2
alpha = .6
optimizer = torch.optim.Adam(student.parameters(),lr = 2e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1,.85,-1)
clip_norm = 7
n_epochs = 10
from random import shuffle 
for epoch in range(n_epochs):
    cum_error = 0
    idx = 0
    t_start = time.time()
    #shuffle(dl_train)
    for inp1,inp2 in pairwise(dl_train): 
        (x1,y1) , (x2,y2)= load_dl(dl_train_type,inp1,inp2) 
        x = torch.stack((x1,x2),0) 
        y = torch.stack((y1,y2),0) 
        optimizer.zero_grad()
        yy = y.to(device)
        xx = x.to(device)
        #with torch.no_grad():
        #    s_teacher = Variable(teacher(yy.view(batch_size,1,-1)),requires_grad = False)
        s_student = student(yy.view(batch_size,1,-1))
        loss_fn_student = loss_by_relative_si_snr(xx,s_student,yy)
        #loss_fn_teacher_student = loss_by_relative_si_snr(s_teacher.permute(1,0,4,2,3).squeeze(4),s_student,yy)
        loss = loss_fn_student#*alpha + (1-alpha)*loss_fn_teacher_student
        cum_error += loss.item()
        error = loss
        error.backward()
        torch.nn.utils.clip_grad_norm_(
                    student.parameters(), clip_norm)
        optimizer.step()
        idx += 1
        if idx > 5000:
            break
        precentile_done = round(100*(idx + 1)/5000)
        progress_symbols = int(np.floor(precentile_done*80/100))
        print('\r['
                + ('#')*progress_symbols
                + (' ')*(80 - progress_symbols)
                + ']' +
                ' Epoch {}/{} progress {}/100% , si_snr_i = {}'.format(epoch + 1, n_epochs, precentile_done,-cum_error/idx), end='')
    t_epoch = measure_epoch_time(t_start)
    print('\n')
    print('*'*33 + 'epoch  results' + '*'*33)
    print('epoch {} , si_snr {:+.2f},si snr improvement {:+.2f}, epoch time {:+.2f} [min], estimated time last {:+.2f} [min]'.format(epoch,-cum_error/idx,0,t_epoch/60,(n_epochs- epoch - 1)*(t_epoch/60)))
    print('*'*80)
    scheduler.step()

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

def check_sparisity(md):
    whole = 0
    nonz = 0
    with torch.no_grad():
        for p in md.parameters():
            x = p.data.detach()
            whole += x.view(-1).numel()
            nonz += x.view(-1).nonzero().numel()
    return nonz/whole

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

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,2,.90,-1)
pruning_factor = .4
score_thd = 8.1
fine_tune_iter = 2
cnt_prune = 0
masks = []
for p in student.seperation_net.parameters():
    if len(p.data.size()) != 1:
        masks.append(gen_mask(p.data.flatten(),pruning_factor))

'''

    Fine Tuning Pruning Stage (pruning and training)

'''
n_epochs = 100
optimizer = torch.optim.Adam(student.parameters(),lr = .5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,2,.90,-1)
for epoch in range(n_epochs):
    #shuffle(dl_train)
    if (cnt_prune > fine_tune_iter):
            if (-cum_error/idx) > score_thd*0: 
                if pruning_factor < .50:
                    cnt_prune = 0
                    #pruning_factor = pruning_factor/.7
                    masks = []
                    for p in student.seperation_net.parameters():
                        if len(p.data.size()) != 1:
                            masks.append(gen_mask(p.data.flatten(),pruning_factor))
    cnt_prune += 1
    t_start = time.time()
    grad_accum = 0
    cum_error = 0
    idx = 0
    si_snr_rel = 0
    si_snr_i_tmp = 0
    cnt = 0
    for p in student.seperation_net.parameters():
        if len(p.data.size()) != 1:
            shape = p.data.shape
            p.data = (p.data.flatten()*masks[cnt]).view(shape)
            cnt += 1
    for inp1,inp2 in pairwise(dl_train): 
        (x1,y1) , (x2,y2)= load_dl(dl_train_type,inp1,inp2) 
        x = torch.stack((x1,x2),0) 
        y = torch.stack((y1,y2),0) 
        optimizer.zero_grad()
        yy = y.to(device)
        xx = x.to(device)
        #cnt = 0
        '''
        for p in student.seperation_net.parameters():
            if len(p.data.size()) != 1:
                shape = p.data.shape
                p.data = (p.data.flatten()*masks[cnt]).view(shape)
                cnt += 1
        '''
        s_student = student(yy.view(batch_size,1,-1))
        loss_fn_student = loss_by_relative_si_snr(xx,s_student,yy)
        #loss_fn_teacher_student = loss_by_relative_si_snr(s_teacher.permute(1,0,4,2,3).squeeze(4),s_student,yy)
        loss = loss_fn_student#*alpha + (1-alpha)*loss_fn_teacher_student
        cum_error += loss.item()
        error = loss
        error.backward()
        torch.nn.utils.clip_grad_norm_(
                    student.parameters(), clip_norm)
        optimizer.step()
        idx += 1
        #cnt = 0
        '''
        for p in student.seperation_net.parameters():
            if len(p.data.size()) != 1:
                shape = p.data.shape
                p.data = (p.data.flatten()*masks[cnt]).view(shape)
                cnt += 1
        '''
        if idx > 1000:
            break
        precentile_done = round(100*(idx + 1)/1000)
        progress_symbols = int(np.floor(precentile_done*80/100))
        print('\r['
                + ('#')*progress_symbols
                + (' ')*(80 - progress_symbols)
                + ']' +
                ' Epoch {}/{} progress {}/100% , si_snr_i = {}'.format(epoch + 1, n_epochs, precentile_done,-cum_error/idx), end='')
    t_epoch = measure_epoch_time(t_start)
    print('\n')
    print('*'*33 + 'epoch  results' + '*'*33)
    print('epoch {} , si_snr {:+.2f},sparisity {:+.2f}, epoch time {:+.2f} [min], estimated time last {:+.2f} [min]'.format(epoch,-cum_error/idx,check_sparisity(student),t_epoch/60,(n_epochs- epoch - 1)*(t_epoch/60)))
    print('*'*80)
    scheduler.step()