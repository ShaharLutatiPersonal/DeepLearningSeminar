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


def si_snr(s,shat):
    s = s.squeeze().permute(0,2,1)
    eps = 1e-8
    shat = shat.squeeze()
    x_zm = shat - torch.mean(shat, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return torch.sum(10 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)),dim = -1)/s.shape[0]

def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

def upit_loss(s,shat):
    '''
    s - BxTxC
    shat BlockxCxBx1xT
    '''
    permute_vec = [[0,1],[1,0]]
    num_blocks = shat.shape[0]
    tot_loss = 0
    for block in range(num_blocks):
        s_hat_block = shat[block,:,:,:,:]
        si_block = torch.tensor([-np.inf]).cuda()
        for perm in permute_vec:
            s_tmp = s_hat_block[perm,:,:,:].squeeze()
            si_block = torch.cat((si_block,torch.sum(si_snr(s.squeeze(),s_tmp)).unsqueeze(0)/2),0)
        tot_loss += torch.max(si_block)
    max_perm = permute_vec[torch.argmax(si_block)-1]
    return -tot_loss/num_blocks,max_perm




class WolfModel(nn.Module):
    def __init__(self,N,L,K,num_blocks,T):
        super(WolfModel, self).__init__()
        self.N,self.L,self.K,self.num_blocks= N,L,K,num_blocks
        simple_net = nn.Conv1d(in_channels=1,out_channels=N,kernel_size=L,stride=L//2)
        xx,ss = self._Segmentation(simple_net(torch.randn(1,1,T)),self.K)
        self.R = xx.shape[3]
        R = self.R
        self.num_speakers = 2
        self.norm = nn.GroupNorm(1, 1, eps=1e-8)
        self.activation = nn.ReLU()
        self.encoding = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=N,kernel_size=L,stride=L//2),nn.ReLU())
        self.seperation_net = SeperationNet(in_channels= N,out_channels = N,hidden_channels = 200,num_layer = num_blocks,num_speakers = 2)
        self.decoding = nn.Sequential(nn.PReLU(),nn.Conv2d(in_channels=R,out_channels=self.num_speakers*R,kernel_size=1))
        self.deconv = nn.Sequential(nn.ConvTranspose1d(in_channels=N,out_channels=1,kernel_size=L,stride=L//2,bias = False))

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
        y = self.encoding(self.norm(input))
        Chunks,gap = self._Segmentation(y,self.K)
        B,N,K,R = Chunks.shape
        new_outputs = [self.seperation_net.blocklist[0](Chunks)]
        for i in range(1,self.num_blocks):
            new_outputs.append(self.seperation_net.blocklist[i](new_outputs[i-1]))
        new_outputs = [self._over_add(self.decoding(x.permute(0,3,2,1)).permute(0,3,2,1).view(B,N,K,R,self.num_speakers),gap) for x in new_outputs]
        deconv_outputs = []
        for x in new_outputs:
            tmp = []
            for ix in range(self.num_speakers):
                tmp.append(self.deconv(x[:,:,:,ix]))
            deconv_outputs.append(torch.stack(tmp))
        return self.activation(torch.stack(deconv_outputs))




class SeperationNet(nn.Module):
    def __init__(self,in_channels,out_channels,hidden_channels,num_layer,num_speakers):
        super(SeperationNet,self).__init__()
        self.blocklist = nn.ModuleList([])
        for i in range(num_layer):
            self.blocklist.append(SepBlock(out_channels, hidden_channels,num_speakers))

class SepBlock(nn.Module):
        def __init__(self,out_channels,hidden_channels,num_speakers):
            super(SepBlock,self).__init__()
            self.gru_1 = nn.LSTM(input_size=out_channels,hidden_size=hidden_channels,num_layers=1,batch_first=True,bidirectional=True,dropout=0)
            self.gru_3 = nn.LSTM(input_size=out_channels,hidden_size=hidden_channels,num_layers=1,batch_first=True,bidirectional=True,dropout=0)
            self.P1 = nn.Linear(hidden_channels*2 ,out_channels,bias=False)
            self.P2 = nn.Linear(hidden_channels*2,out_channels,bias=False)
            self.norm1 = nn.GroupNorm(1, out_channels, eps=1e-8)
            self.norm2 = nn.GroupNorm(1, out_channels, eps=1e-8)
        def forward(self,input):
            B,N,K,R = input.shape
            short_term = input.permute(0,3,2,1).contiguous().view(B*R, K, N)
            short_term1,_  = self.gru_1(short_term)
            intra_rnn = self.norm1(self.P1(short_term1.contiguous().view(B*R*K,-1)).view(B*R,K,-1).view(B,R,K,N).permute(0,3,2,1))# + input
            transition_tensor = intra_rnn.permute(0,2,3,1).contiguous().view(B*K,R,N)
            long_term1,_s3 = self.gru_3(transition_tensor)
            return self.norm2(self.P2(long_term1.contiguous().view(B*R*K, -1)).view(B*K, R, -1).view(B,K,R,N).permute(0,3,1,2).contiguous())# + intra_rnn
            
            
             #transition_tensor).permute(0,2,1).contiguous().view(B,K,N,R).permute(0,3,2,1) 
            