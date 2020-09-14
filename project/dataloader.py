import torch
from torch.utils import data
import librosa
import soundfile as sf
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from handle_folder import db_assignment_problem_solver 
from torch.utils.data.sampler import RandomSampler



def read_audio(fname,target_sr,T):
    data,sr = sf.read(fname)
    data = data.T
    data = librosa.resample(data,sr,target_sr)
    len_allowed = int(T * np.ceil((len(data)/T)))
    data = np.concatenate([data,np.zeros([len_allowed-len(data)])],axis=0)
    data = data/np.max(data)
    return data

def mix_audio(S):
    '''
    S dimensions TxC
    '''
    mix_waves_factor = torch.rand(2)
    mix_waves_factor = mix_waves_factor/sum(mix_waves_factor)
    for i in range(S.shape[1]):
        S[:,i] = S[:,i]*mix_waves_factor[i]
    res = torch.sum(S,dim = 1)
    return res

    

class AudioDataset(data.Dataset):
    'Dataset comment'
    def __init__(self,path,batch_size = 1,T = 2.5,target_sr = 8e3,is_processed = False):
        self.batch_size = batch_size
        self.T = int(T*target_sr)
        self.target_sr = target_sr
        self.pairs = db_assignment_problem_solver(path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self,index):
        audio =  []
        pair = self.pairs[index]
        for p in pair:
            audio.append(torch.tensor(read_audio(p,self.target_sr,self.T)))
        min_len = min([len(x) for x in audio])
        audio = [x[:min_len] for x in audio]
        S = torch.stack(audio,dim = 1)
        mixed_audio = mix_audio(S)
        S = S.view(-1,self.T,2)
        mixed_audio = mixed_audio.view(-1,self.T)
        return S,mixed_audio



class DataLoader(object):

    def __init__(self, dataset = AudioDataset, batch_size = 10, drop_last=True):
        self.ds = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = RandomSampler(dataset)

    def __iter__(self):
        batch_mix = torch.Tensor()
        batch_orig = torch.Tensor()
        
        for idx in self.sampler:
            print("idx")
            print(idx)
            orig,mix = self.ds[idx]
            print("orig shape")
            print(orig.shape)
            print("mix shape")
            print(mix.shape)
            batch_mix = torch.cat([batch_mix, mix.float()],dim = 0)
            batch_orig = torch.cat([batch_orig, orig.float()],dim = 0)
            while batch_mix.size(0) >= self.batch_size:
                if batch_mix.size(0) == self.batch_size:
                    yield batch_orig,batch_mix
                    batch_mix = torch.Tensor()
                    batch_orig = torch.Tensor()
                else:
                    return_batch_orig, batch_orig = batch_orig.split([self.batch_size,batch_orig.size(0)-self.batch_size],dim = 0)
                    return_batch_mix, batch_mix = batch_mix.split([self.batch_size,batch_mix.size(0)-self.batch_size],dim = 0)
                    yield return_batch_orig,return_batch_mix
        if batch_mix.size(0) > 0 and not self.drop_last:
            yield batch_orig,batch_mix