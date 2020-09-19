import torch
import numpy as np
import sparsified_model
import soundfile as sf
import librosa
import scipy.io.wavfile as wave
import argparse

def read_audio(fname,target_sr = 8000,T = 4, starting_point = 0):
    data,sr = sf.read(fname)
    data = data.T
    data = librosa.resample(data,sr,target_sr)
    len_allowed = T*target_sr
    valid = True if len(data)>len_allowed else False
    if valid:
        data = data[starting_point*target_sr:starting_point*target_sr + (target_sr*T)]
        data = data/np.max(data)
    return data,valid

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)


def main_func(file_path,dict_path,starting_point):
    print('Start by loading file, and resample it')
    data , _ = read_audio(file_path,starting_point=starting_point)
    print('finished to resample')
    if not _:
        print("Error ! not valid audio length !!!")
        return 0
    print('some memory allocation')
    data = torch.tensor(data,dtype=torch.float)
    #zeros_data = torch.zeros(data.shape,dtype=torch.float)
    data = [data.unsqueeze(0)]
    dictionary = torch.load(dict_path)
    device = 'cpu'
    T = 4 # seconds
    model = sparsified_model.WolfModel(64,16,256,3,int(T*8e3),multi_loss = False, hidden_size = 128 ,bidirectional = True,MulCat = True,weights_for_seperation=dictionary['sparsed_dict'])
    model.load_my_state_dict(dictionary['origin_dict'])
    model = model.to(device)
    model.eval()
    print('now the magic is happening')
    with torch.no_grad():
        cnt = 0
        for inp1 in data:
            inp = inp1
            inp = inp.to(device)
            out = model(inp.view(1,1,-1)).detach()
            sig = out
            for batch in [0]:
                sig_1 = sig[batch,0,:].squeeze().cpu().numpy()
                sig_2 = sig[batch,1,:].squeeze().cpu().numpy()
                sig_1 = sparsified_model.clear_sound(sig_1)
                sig_2 = sparsified_model.clear_sound(sig_2)
                wave.write('{} {} 1.wav'.format(cnt,batch),8000,sig_1)
                wave.write('{} {} 2.wav'.format(cnt,batch),8000,sig_2)
            cnt += 1
    print('done')
    return 0






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file_path', type=str, default='test.flac',
        help='path to audio file test.flac' )
    parser.add_argument(
        '-d', '--dict_path', type=str, default='sparsed_dict_to_use.pth', help='sparsed dictionary file path')
    parser.add_argument(
        '-t', '--starting_point', type=int, default=0, help='starting point of recording, in seconds')
    args = parser.parse_args()

    main_func(args.file_path, args.dict_path,args.starting_point)