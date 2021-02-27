from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from random import randint

## define size of the input data
time_length = 10039 ## 10039 is half of the sector time
mid_length = (time_length - 1)//2
num_channels = 12


def select_transits(transits):
    if len(transits)<=num_channels:
        return transits
    else:
        start = randint(0,len(transits)-num_channels)
        return transits[start:start+num_channels]
    
class FLCDataset(Dataset):
    def __init__(self, npzfile = 'model_input/train.npz', mode='vetting'):
        data = np.load(npzfile,allow_pickle=True)
        self.lc = data['lc']
        self.transits = data['transits']
        self.period = data['period']
        self.tic = data['tic']
        dispositions = data['dispositions']
        
        pc = np.array(['PC']*len(self.lc))
        eb = np.array(['EB']*len(self.lc))

        if mode == 'triage':
            self.y = np.array(np.logical_or(dispositions==eb, dispositions==pc), dtype=int)
        elif mode == 'vetting':
            self.y = np.array(dispositions==pc, dtype=int)
        print(npzfile,' size and number of positives of data:', dispositions.shape[0], np.sum(self.y))
    
    def __getitem__(self, index):
        flc = np.ones((num_channels, time_length)) ## phase folded into channels
        ts = self.transits[index]
        hp = int(np.clip(self.period[index]//2,a_min=0,a_max=mid_length)) ## half period
        #print('half period ', hp)
        #xxx=input()
        sts = select_transits(ts)

        for c, t in enumerate(sts):
            t = int(t)
            flc[c, mid_length-hp:mid_length+hp+1] = np.clip(self.lc[index],a_min=0.0,a_max=1.01)[t-hp:t+hp+1]

        ##normalization
        depth = np.min(flc)
        flc = (flc-1.0)/(1.0 - depth + 1e-8)
    
        return self.tic[index], np.asarray(flc,dtype=np.float32), np.asarray([self.y[index]],dtype=np.float32)

    def __len__(self):
        return self.y.shape[0]

