# An Input Representation for Convolutional Neural Networks in Exoplanet Transit Detection

## About this repo
The TESS light curves are preprocessed with each transit phase placed as channels of the neural network input. And a neural network model is trained to detect signals of exoplanet with TESS 2-min cadence data.

## List of python files for this project
[data_preprocess](data_preprocess/)
- [preprocess.py](data_preprocess/preprocess.py) downloading, detrending the light curves for the training-validation-test dataset.
- [toi_preprocess.py](data_preprocess/toi_preprocess.py) downloading, detrending the light curves for the TOI dataset.
- [process_lightcurve_with_two_cadence.py](data_preprocess/process_lightcurve_with_two_cadence.py) represent the light curve as an input representation.

[models](models/)
- [augment.py](models/augment.py) data augmentation methods
- [flcdataset.py](models/flcdataset.py) phase segmenting the light curves into channels for model training and referring.
- [models.py](models/models.py) neural network architecture
- [train_model.py](models/train_model.py) model training and referring 
- [test_results.py](models/test_results.py) averaging the cross validation results for the test set and show some results.

[toi_results](tois_results/)
- [toi_results.py](tois_results/toi_results.py) averaging the cross validation results for the TOIs and show some results.


## Preparing the data for the neural network
Run ```python data_preprocess/preprocess.py``` to get 'train_80.npz' and 'test_20.npz' in the 'model_input' folder.

Run ```python data_preprocess/toi_preprocess.py``` to get 'tois.npz' in the 'model_input' folder.

The above npz files can be obtained from [here](https://www.jianguoyun.com/p/DRlLOrUQ2J2gCRiE2OMD).
## Model training and referring
For example, run
```
python train_model.py plain vetting irs 3
```
to train a *plain* model for the exoplanet *vetting* task with data augmentation method inversion (*i*), rescale (*r*) and channel shuffle (*s*) on the *3rd*-fold data as the validation set.

To train models for the 5-fold cross validation and five data augmentation methods, run the following commands

```
python train_model.py plain vetting none 0
python train_model.py plain vetting none 1
python train_model.py plain vetting none 2
python train_model.py plain vetting none 3
python train_model.py plain vetting none 4
python train_model.py plain vetting i 0
python train_model.py plain vetting i 1
python train_model.py plain vetting i 2
python train_model.py plain vetting i 3
python train_model.py plain vetting i 4
python train_model.py plain vetting r 0
python train_model.py plain vetting r 1
python train_model.py plain vetting r 2
python train_model.py plain vetting r 3
python train_model.py plain vetting r 4
python train_model.py plain vetting s 0
python train_model.py plain vetting s 1
python train_model.py plain vetting s 2
python train_model.py plain vetting s 3
python train_model.py plain vetting s 4
python train_model.py plain vetting irs 0
python train_model.py plain vetting irs 1
python train_model.py plain vetting irs 2
python train_model.py plain vetting irs 3
python train_model.py plain vetting irs 4
```

run
```
python test_results.py
```
to average the five results for the test set, including accuracy, precision, recall, AUC, APS, and plot the precision, recall curve plot for results with different augmentation methods.

run
```
python toi_results.py
```
average the reuslts and plot the model prediction for 2-min cadence known planet TOIs against signal-to-noise ratios  and transit depths.
