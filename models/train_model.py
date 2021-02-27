import argparse

import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from random import random

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models import *
from augment import *
from flcdataset import *

parser = argparse.ArgumentParser()
parser.add_argument('model', help='model name', type=str)
parser.add_argument('mode', help='triage or vetting', type=str)
parser.add_argument('aug', help='data augmentation method', type=str)
parser.add_argument('k', help='the k-th fold', type=int)
args = parser.parse_args()

## read the 80% training data
data = np.load('../model_input/train_80.npz',allow_pickle=True)
tic = np.asarray(data['tic'])
lc = np.asarray(data['lc'])
transits = np.asarray(data['transits'])
period = np.asarray(data['period'])
dispositions = data['dispositions']

n_data = tic.shape[0]
dice = np.arange(n_data)
n_valid = int(n_data/5.0) ## number of validation set, 5-fold

valid_index = dice[args.k*n_valid : (args.k+1)*n_valid]
train_index = list(set(dice)-set(valid_index))

np.savez('../model_input/train_kfold.npz', tic=tic[train_index], lc=lc[train_index], transits=transits[train_index], period=period[train_index], dispositions=dispositions[train_index])

np.savez('../model_input/val_kfold.npz', tic=tic[valid_index], lc=lc[valid_index], transits=transits[valid_index], period=period[valid_index], dispositions=dispositions[valid_index])

torch.backends.cudnn.benchmark = True

train = FLCDataset('../model_input/train_kfold.npz', mode=args.mode)
trainloader = DataLoader(train, shuffle=True, num_workers=1, batch_size=64)
val = FLCDataset('../model_input/val_kfold.npz', mode=args.mode)
valloader = DataLoader(val, shuffle=False, num_workers=1, batch_size=64)
test = FLCDataset('../model_input/test_20.npz', mode=args.mode)
testloader = DataLoader(test, shuffle=False, num_workers=1, batch_size=64)

if args.model == 'plain':
    net = ModelPlain(n=32).cuda() ## plain convolutional network
else:
    net = ModelInTest(n=32).cuda() ## Residual Network

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('This model has', total_params, 'parameters.')

criterion = nn.BCELoss().cuda()

optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

train_log = open('./output/training_record_'+str(args.model)+'_'+str(args.aug)+'_'+str(args.k)+'.log','w')
train_log.write('epoch,train_loss,test_loss,test_aps,test_auc\n')

print('start training model '+str(args.model)+' augmentation '+str(args.aug)+' fold '+str(args.k))
best_loss_val = 200.0
tolerence = 0
for epoch in range(1,101,1):
    net.train()
    loss_train = 0.0
    #num_corr_train, num_train = 0.0, 0.0
    for (_, flc, targets) in trainloader:
        flc = flc.cuda()
        targets = targets.cuda()

        if 'i' in args.aug:
            if random()<0.5:
                flc = invert(flc)
        if 'r' in args.aug:
            if random()<0.9:
                cut = np.random.randint(-1000,1000)
                flc = hscale(flc, 10039, cut)
        if 's' in args.aug:
            if random()<0.9:
                inputs = channel_shuffle(flc)
        
        outputs = net(flc)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_train += loss.item()*targets.size(0)

    ## results on the validataion set
    net.eval()

    loss_val = 0.0
    prediction, groundtruth = [], []
    for (_, flc, targets) in valloader:
        flc = flc.cuda()
        targets = targets.cuda()
        outputs = net(flc)
        loss = criterion(outputs, targets)
        loss_val += loss.item()*targets.size(0)
        prediction.append(outputs.data.cpu().numpy())
        groundtruth.append(targets.data.cpu().numpy())
    prediction = np.concatenate(prediction).flatten()
    groundtruth = np.concatenate(groundtruth).flatten()
    aps_val = average_precision_score(groundtruth, prediction)
    auc_val = roc_auc_score(groundtruth, prediction)

    loss_test = 0.0
    prediction, groundtruth = [], []
    for (_, flc, targets) in testloader:
        flc = flc.cuda()
        targets = targets.cuda()
        outputs = net(flc)
        loss = criterion(outputs, targets)
        loss_test += loss.item()*targets.size(0)
        prediction.append(outputs.data.cpu().numpy())
        groundtruth.append(targets.data.cpu().numpy())
    prediction = np.concatenate(prediction).flatten()
    groundtruth = np.concatenate(groundtruth).flatten()
    aps_test = average_precision_score(groundtruth, prediction)
    auc_test = roc_auc_score(groundtruth, prediction)


    print(epoch, ' {0:.1f}'.format(loss_train), ' {0:.1f}'.format(loss_val),' {0:.1f}'.format(loss_test),\
          '| {0:.4f}'.format(aps_val), \
          ' {0:.4f}'.format(aps_test), \
          '| {0:.4f}'.format(auc_val), \
          ' {0:.4f}'.format(auc_test))

    if loss_val < best_loss_val:
        best_loss_val = loss_val
        torch.save(net.state_dict(), './output/model'+str(args.model)+'_'+str(args.aug)+'_'+str(args.k)+'.pt')
        print('model saved...')
        tolerence = 0
    tolerence += 1
    if tolerence>30:
        break

    train_log.write('{:d},{:.5f},{:.5f},{:.5f},{:.5f}\n'\
                    .format(epoch, loss_train, loss_test, aps_test, auc_test))

train_log.close()

#######################
## training finished ##
#######################

## load the model parameters with the best performance on the validation set
net.load_state_dict(torch.load('./output/model'+str(args.model)+'_'+str(args.aug)+'_'+str(args.k)+'.pt'))
net.eval()

pred_test = [] ## model prediction and ground truth for the test set
for (_, flc, targets) in testloader:
    flc = flc.cuda()
    targets = targets
    outputs = net(flc)
    pred_test.append(outputs.data.cpu().numpy())
    
pred_test = np.concatenate(pred_test).flatten()
np.save('./output/test_prediction_kfold_'+str(args.model)+'_'+str(args.aug)+'_'+str(args.k)+'.npy', pred_test)


## output the prediction for TOIs
tois = FLCDataset('../model_input/tois.npz', mode=args.mode)
toiloader = DataLoader(tois, shuffle=False, num_workers=1, batch_size=64)

pred_toi = [] ## model prediction and ground truth for the toi set
for (_, flc, targets) in toiloader:
    flc = flc.cuda()
    targets = targets
    outputs = net(flc)
    pred_toi.append(outputs.data.cpu().numpy())
    
pred_toi = np.concatenate(pred_toi).flatten()
np.save('./output/toi_prediction_kfold_'+str(args.model)+'_'+str(args.aug)+'_'+str(args.k)+'.npy', pred_toi)
