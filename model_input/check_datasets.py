import numpy as np
import matplotlib.pyplot as plt

num_channels = 12
mid_length = 10039//2
def select_transits(transits):
    if len(transits)<=num_channels:
        return transits
    else:
        start = np.random.randint(0,len(transits)-num_channels)
        return transits[start:start+num_channels]


data = np.load('test_20.npz',allow_pickle=True)
lc = data['lc']
transits = data['transits']
#y = data['vetting_y']
d = data['dispositions']
period = data['period']
print(len(d))
print(np.sum(d=='PC'))

for i in range(len(period)):
    flc = np.ones((12, 10039))
    ts = transits[i]
    hp = int(period[i]//2) ## half period
    sts = select_transits(ts)

    c = 0
    for t in sts:
       t = int(t)
       flc[c, mid_length-hp:mid_length+hp+1] = lc[i][t-hp:t+hp+1]
       c += 1
    eps = 1e-8
    depth = np.min(flc)
    flc = (flc-1.0)/(1.0 - depth + eps)
    
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, top=0.95, right=0.90, bottom=0.05, wspace=0.15, hspace=0.35)
    plt.subplot(2, 1, 1)
    plt.title(d[i])
    plt.plot(np.arange(len(flc[0])), flc[0], '.', color='r', markersize=2.5)
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(flc[1])), flc[1], '.', color='g', markersize=2.5)
    #if d[i] == 'KP':
    plt.show()
    plt.close()

