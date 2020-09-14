import dataloader
import torch
import numpy as np
base = r'C:\Users\shaha\Documents\DeepGit\DeepLearningSeminar\project\dataraw\train-clean-100'
batch_size = 1
T = 4
target_sr = 8e3
ds_train = dataloader.AudioDataset(path = base,batch_size = batch_size,T = T,target_sr = target_sr)
dl_train = dataloader.DataLoader(dataset=ds_train,batch_size = batch_size)
len_t = len(ds_train)
for idx,(x,y) in enumerate(dl_train):
    data = (x,y)
    torch.save(data,r'C:\Users\shaha\Documents\DeepGit\DeepLearningSeminar\project\data\{}.pt'.format(idx))
    precentile_done = round(100*(idx + 1)/len_t)
    progress_symbols = int(np.floor(precentile_done*80/100))
    print('\r['
                  + ('#')*progress_symbols
                  + (' ')*(80 - progress_symbols)
                  + ']' +
                  'progress {}/100%'.format( precentile_done), end='')
print('*'*80)
print('finished !')