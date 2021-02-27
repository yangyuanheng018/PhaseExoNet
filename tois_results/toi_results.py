import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.cm as cm
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc

## load tois.csv
tois_csv = pd.read_csv('../target_info/tois.csv', header=0, skiprows=4) 

## load TOI tic id and dispositions
data = np.load('../model_input/tois.npz', allow_pickle=True)
toi_tic = data['tic']
toi_dispositions = data['dispositions']
toi_lc = data['lc']
toi_transits = data['transits']

## load and average the model prediction for TOIS
toi_prediction = np.load('../models/output/toi_prediction_kfold_plain_irs_0.npy')
for i in range(1,5,1):
    toi_prediction += np.load('../models/output/toi_prediction_kfold_plain_irs_'+str(i)+'.npy')
toi_prediction /= 5.0

## load TOI informations
toi_info = np.load('../model_input/valid_spoc_toi_info.npz')
toi_tmag = toi_info['tmag']
toi_period = toi_info['period']
toi_duration = toi_info['duration']
toi_depth = toi_info['depth']
toi_snr = toi_info['snr']

## TOI indices for known planets and others
kp_indices = np.where(toi_dispositions=='KP')
#other_indices = np.where(toi_dispositions!='KP')

## show the number of recovered KP at several thresholds
num_kp = len(kp_indices[0])
for threshold in (0.1, 0.25, 0.3, 0.5):
    num_recovered = np.sum(np.array(toi_prediction[kp_indices]>threshold, dtype=int))
    print(threshold, num_recovered, num_kp, num_recovered/float(num_kp))


## draw the scatter plot for the KPs
#sc = plt.scatter(np.log(toi_snr[kp_indices]), np.log(toi_depth[kp_indices]), c=toi_prediction[kp_indices], cmap=plt.cm.RdYlBu) #cm.get_cmap('RdYlBu'))
sc = plt.scatter(toi_snr[kp_indices], toi_depth[kp_indices], c=toi_prediction[kp_indices], cmap=plt.cm.RdYlBu)
plt.colorbar(sc, label='Prediction')
plt.semilogx()
plt.semilogy()
plt.xlim(5.9,1000)
plt.xlabel('SNR')
plt.ylabel('Transit Depth (ppm)')
plt.show()

## histogram of the prediction
plt.hist(toi_prediction[kp_indices], bins=40, color='darkseagreen')
plt.show()

## show the light curves of the missed KPs
for c, t, p, l, s, d in zip(toi_tic[kp_indices], toi_dispositions[kp_indices], toi_prediction[kp_indices], toi_lc[kp_indices], toi_transits[kp_indices], toi_depth[kp_indices]):
    if t=='KP' and p<0.5 and d>10000:
        plt.plot(np.arange(len(l)), l, '.', color='lightseagreen')
        plt.plot(s, np.ones(len(s))*np.min(l), 'r^')
        if c == 432549364:
            plt.title('TIC '+str(c)+ ' (KELT-1b)')
        else:
            plt.title('TIC '+str(c))
        plt.show()
        print(c, p)

## save prediction of tois
for i in range(len(toi_period)):
    tois_csv.loc[tois_csv['Orbital Period Value'] == toi_period[i],'Prediction'] = '%.5f' % toi_prediction[i]
tois_csv.dropna(subset=['Prediction'],inplace=True)
tois_csv.to_csv('tois_prediction.csv')
print(len(tois_csv))

        




