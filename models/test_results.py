import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc

from sklearn.metrics import accuracy_score

#y = [1,0,1,0]
#t = [1,1,0,0]
#print(precision_score(t,y))
#print(recall_score(t,y))
#y = [0.8,0.2,0.9,0.5]
#t = [1,1,0,0]
#print(precision_score(t,y))
#print(recall_score(t,y))

def scores(t, model='plain', aug='none', threshold=0.5):
    y = np.zeros((5,len(t)))
    for i in range(5):
        data = np.load('./output/test_prediction_kfold_'+model+'_'+aug+'_'+str(i)+'.npy')
        assert len(data) == len(t)
        y[i] = data

    y_mean = np.mean(y,axis = 0)
    y_binarized = np.array(y_mean > threshold, dtype=float)
    print(model+'_'+aug+'_AUC_APS:')
    print(roc_auc_score(t,y_mean))
    print(average_precision_score(t,y_mean))
    print('threshold', threshold,'recovered', np.sum(y_binarized*t), '/', np.sum(t), \
          'accuracy {0:.4f}'.format(accuracy_score(t, y_binarized)), \
          'precision {0:.4f}'.format(precision_score(t, y_binarized, zero_division=1)),'recall {0:.4f}'.format(recall_score(t, y_binarized)))

## load ground truth label for vetting
test_dispositions = np.load('../model_input/test_20.npz')['dispositions']
pc = np.array(['PC']*len(test_dispositions))
#eb = np.array(['EB']*len(test_dispositions))
t = np.array(test_dispositions==pc, dtype=int)

scores(t, 'plain', 'none')
scores(t, 'plain', 'i')
scores(t, 'plain', 'r')
scores(t, 'plain', 's')
scores(t, 'plain', 'irs')

scores(t, 'plain', 'none', 0.25)
scores(t, 'plain', 'i', 0.25)
scores(t, 'plain', 'r', 0.25)
scores(t, 'plain', 's', 0.25)
scores(t, 'plain', 'irs', 0.25)

y_irs = np.zeros((5,len(t)))
for i in range(5):
    data = np.load('./output/test_prediction_kfold_plain_irs_'+str(i)+'.npy')
    assert len(data) == len(t)
    y_irs[i] = data
y_mean_irs = np.mean(y_irs,axis = 0)

y_none = np.zeros((5,len(t)))
for i in range(5):
    data = np.load('./output/test_prediction_kfold_plain_none_'+str(i)+'.npy')
    assert len(data) == len(t)
    y_none[i] = data
y_mean_none = np.mean(y_none,axis = 0)

y_i = np.zeros((5,len(t)))
for i in range(5):
    data = np.load('./output/test_prediction_kfold_plain_i_'+str(i)+'.npy')
    assert len(data) == len(t)
    y_i[i] = data
y_mean_i = np.mean(y_i,axis = 0)

y_r = np.zeros((5,len(t)))
for i in range(5):
    data = np.load('./output/test_prediction_kfold_plain_r_'+str(i)+'.npy')
    assert len(data) == len(t)
    y_r[i] = data
y_mean_r = np.mean(y_r,axis = 0)

y_s = np.zeros((5,len(t)))
for i in range(5):
    data = np.load('./output/test_prediction_kfold_plain_s_'+str(i)+'.npy')
    assert len(data) == len(t)
    y_s[i] = data
y_mean_s = np.mean(y_s,axis = 0)

## For irs augmentation results, show the scores under several thresholds
for threshold in (0.1, 0.2, 0.25, 0.5):
    binarized_y = np.array(y_mean_irs > threshold, dtype=float)
    print('threshold', threshold,'recovered', np.sum(binarized_y*t), '/', np.sum(t), \
          'precision {0:.4f}'.format(precision_score(t, binarized_y, zero_division=1)),'recall {0:.4f}'.format(recall_score(t, binarized_y)))

xxx=input('press enter to draw the thershold-precision/recall curve')

thresholds = np.arange(0,1,0.001)

precision_irs, recall_irs = [], []
for thres in thresholds:
    binarized_y = np.array(y_mean_irs > thres, dtype=float)
    precision_irs.append(precision_score(t, binarized_y, zero_division=1))
    recall_irs.append(recall_score(t, binarized_y))

precision_none, recall_none = [], []
for thres in thresholds:
    binarized_y = np.array(y_mean_none > thres, dtype=float)
    precision_none.append(precision_score(t, binarized_y, zero_division=1))
    recall_none.append(recall_score(t, binarized_y))

precision_i, recall_i = [], []
for thres in thresholds:
    binarized_y = np.array(y_mean_i > thres, dtype=float)
    precision_i.append(precision_score(t, binarized_y, zero_division=1))
    recall_i.append(recall_score(t, binarized_y))

precision_r, recall_r = [], []
for thres in thresholds:
    binarized_y = np.array(y_mean_r > thres, dtype=float)
    precision_r.append(precision_score(t, binarized_y, zero_division=1))
    recall_r.append(recall_score(t, binarized_y))

precision_s, recall_s = [], []
for thres in thresholds:
    binarized_y = np.array(y_mean_s > thres, dtype=float)
    precision_s.append(precision_score(t, binarized_y, zero_division=1))
    recall_s.append(recall_score(t, binarized_y))

plt.xlim((0,1))
plt.ylim((0,1))
plt.plot(thresholds, precision_irs, '-', color='navy', linewidth=2, label='Precision-irs')
plt.plot(thresholds, recall_irs, '-', color='deeppink', linewidth=2, label='Recall-irs')
plt.plot(thresholds, precision_none, '--', color='green', linewidth=1, label='Precision-none')
plt.plot(thresholds, recall_none, '--', color='pink', linewidth=1, label='Recall-none')
plt.plot(thresholds, precision_i, '--', color='lawngreen', linewidth=1, label='Precision-i')
plt.plot(thresholds, recall_i, '--', color='plum', linewidth=1, label='Recall-i')
plt.plot(thresholds, precision_r, '--', color='cyan', linewidth=1, label='Precision-r')
plt.plot(thresholds, recall_r, '--', color='tomato', linewidth=1, label='Recall-r')
plt.plot(thresholds, precision_s, '--', color='olive', linewidth=1, label='Precision-s')
plt.plot(thresholds, recall_s, '--', color='salmon', linewidth=1, label='Recall-s')
plt.xlabel('Threshold')
plt.legend(loc='best')
plt.xlim
plt.show()

