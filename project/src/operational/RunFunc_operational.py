import torch
import sparsified_model
import soundfile as sf
import librosa
import scipy.io.wavfile as wave
path = 'sparsed_dict_to_use.pth'

def read_audio(fname,target_sr = 8000,T = 4):
    data,sr = sf.read(fname)
    data = data.T
    data = librosa.resample(data,sr,target_sr)
    len_allowed = T
    valid = True if len(data)>len_allowed else False
    if valid:
        data = data[:len_allowed]
        data = data/np.max(data)
    return data,valid

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

audio_path = []
audio = []
for p in audio_path:
    data,_ = read_audio(p,target_sr = 8000,T = 4)
    audio.append(torch.tensor(data))


dictionary = torch.load(path)
device = 'cpu'
T = 4 # seconds
model = sparsified_model.WolfModel(64,16,256,3,int(T*8e3),multi_loss = False, hidden_size = 128 ,bidirectional = True,MulCat = True,weights_for_seperation=dictionary['sparsed_dict'])
model.load_my_state_dict(dictionary['origin_dict'])
model = model.to(device)
model.eval()
with torch.no_grad():
    cnt = 0
    for inp1,inp2 in pairwise(audio)
        inp = torch.stack((inp1,inp2),0)
        inp = inp.to(device)
        out = model(inp.view(2,1,-1)).detach()
        sig = out.squeeze()
        for batch in range(2):
            sig_1 = sig[batch,1,:].squeeze().cpu().numpy()
            sig_2 = sig[batch,1,:].squeeze().cpu().numpy()
            sig_1 = sparsified_model.clear_sound(sig_1)
            sig_2 = sparsified_model.clear_sound(sig_2)
            wave.write('{} {} 1.wav'.format(cnt,batch),8000,sig_1)
            wave.write('{} {} 2.wav'.format(cnt,batch),8000,sig_2)
        cnt += 1

