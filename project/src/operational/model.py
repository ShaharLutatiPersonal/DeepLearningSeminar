import torch
import torch.nn as nn


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
                t_tmp = torch.zeros(s_tmp.shape).to('cpu')
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
            self.rnn_1 = nn.LSTM(input_size=out_channels,hidden_size=hidden_channels,num_layers=1,batch_first=True,bidirectional=True)
            if self.MulCat:
                self.rnn_2 = nn.LSTM(input_size=out_channels,hidden_size=hidden_channels,num_layers=1,batch_first=True,bidirectional=True)
                self.rnn_4 = nn.LSTM(input_size=out_channels,hidden_size=hidden_channels,num_layers=1,batch_first=True,bidirectional=self.bidirectional)
            self.rnn_3 = nn.LSTM(input_size=out_channels,hidden_size=hidden_channels,num_layers=1,batch_first=True,bidirectional=self.bidirectional)
            self.gn1 = nn.GroupNorm(num_groups=1,num_channels=out_channels,eps=1e-8,affine=True)
            self.gn2 = nn.GroupNorm(num_groups=1,num_channels=out_channels,eps=1e-8,affine=True)
            self.P1 = nn.Sequential(nn.Linear(hidden_channels*2 + out_channels,out_channels))
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
        simple_net = nn.Conv1d(in_channels=1,out_channels=N,kernel_size=L,stride=L//2)
        xx,ss = self._Segmentation(simple_net(torch.randn(1,1,T)),self.K)
        self.R = xx.shape[3]
        R = self.R
        self.encoding = nn.Conv1d(in_channels=1,out_channels=N,kernel_size=L,stride=L//2,bias = False)
        self.bn = nn.BatchNorm1d(N,eps = self.eps)
        self.ReLU = nn.ReLU()
        self.deconv = nn.Sequential(nn.ConvTranspose1d(in_channels=N,out_channels=1,kernel_size=L,stride=L//2,bias = False))

    def forward(self,input):
        y = self.ReLU(self.bn(self.encoding(input)))
        #print('shape before')
        #print(y.shape)
        Chunks,gap = self._Segmentation(y,self.K)
        y = self.deconv(self._over_add(Chunks,gap))
        return y
    
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