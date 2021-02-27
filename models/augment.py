import glob
import os
import numpy as np
from scipy.fftpack import fft, ifft
import torch
import torch.nn as nn
import random as rd
from torch.nn.functional import interpolate


def invert(tensor):
    idx = [i for i in range(tensor.size(2)-1, -1, -1)]
    idx = torch.cuda.LongTensor(idx)
    inverted_tensor = tensor.index_select(2, idx)
    return inverted_tensor

def hscale(tensor, s, cut):
    if cut>0:
        hs_tensor = tensor[:,:,cut:-cut]
        hs_tensor = interpolate(hs_tensor, size=s, mode='linear', align_corners=True)
        return hs_tensor
    elif cut<0:
        padding = torch.zeros(tensor.size(0), tensor.size(1), -cut).cuda()
        hs_tensor = torch.cat([padding, tensor, padding], dim=2)
        hs_tensor = interpolate(hs_tensor, size=s, mode='linear', align_corners=True)
        return hs_tensor
    else:
        return tensor
        
def channel_shuffle(tensor):
    '''
    shuffle the even and odd channels separately
    '''
    rng = np.random.default_rng()
    idx = np.arange(8).reshape((2, 4), order='F')
    rng.shuffle(idx, axis=1)
    idx = idx.flatten(order='F')
    shuffled_idx = torch.cuda.LongTensor(idx)
    return tensor.index_select(1, shuffled_idx)

def channel_erase(tensor):
    pass

