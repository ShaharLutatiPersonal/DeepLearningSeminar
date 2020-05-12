import torch
import numpy as np



def si_snr(s,shat):
    si_tild = np.sum(s*shat,axis = 1)*s/np.sum(s*s,axis = 1)
    ei = shat-si_tild
    return np.sum(20*np.log10(np.linalg.norm(si_tild,axis = 1)/np.linalg.norm(ei,axis = 1)))

def upit_loss(s,shat):
    permute_vec = [[0,1],[1,0]]
    min_loss = np.inf
    for perm in permute_vec:
        s_tmp = shat[perm,:]
        loss = si_snr(s,s_tmp)
        if loss > min_loss:
            min_loss = loss
            chosen_perm = perm
    return min_loss,shat[chosen_perm,:],chosen_perm



    