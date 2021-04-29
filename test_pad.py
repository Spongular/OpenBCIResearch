
import mne
from mne.io import read_raw_edf

import standard_classifiers
import data_loading
import gen_tools
import keras_classifiers
from sklearn.utils import shuffle
import numpy as np
from keras import models
from mne.datasets.eegbci import eegbci

mne.set_log_level('WARNING')

#load our data.
raw = data_loading.get_all_mi_between(1, 21, 4, ["088", "092", "100"])
raw = gen_tools.mifixes_and_bandpass(raw, min=6., max=30.)

#classify it.
#standard_classifiers.csp_lda(raw, -1., 4.)
#standard_classifiers.csp_svm(raw, -1., 4.)
#standard_classifiers.csp_lda(raw, -1., 4., ["C3", "Cz", "C4"])

#standard_classifiers.pca_lda(raw, -1., 4., ["C3", "Cz", "C4"], n_jobs=2, pca_n_components=32)
#standard_classifiers.pca_svm(raw, -1., 4., ["C3", "Cz", "C4"], n_logspace=2, n_jobs=2, pca_n_components=32)
#standard_classifiers.pca_knn(raw, -1., 4., ["C3", "Cz", "C4"], max_n_neighbors=10, n_jobs=2, pca_n_components=32)

#Get Epochs
data, labels, epochs = gen_tools.epoch_data(raw, tmin=-1, tmax=1, pick_list=[])
del data, labels #We don't need these.

#Generate Wavelets from Epochs.
X, Y = gen_tools.wavelet_transform(epochs, event_names=epochs.event_id,
                                   f_low=6., f_high=30., )

#Reshape X for FF network
X = gen_tools.reshape_4to3(X)

#Form the Neural Network.
nn, opt = keras_classifiers.feedforward_nn(X.shape, n_classes=2, model_shape=[2000, 1000, 500, 250], d_rate=0.5, b_norm=False)
#nn, opt = keras_classifiers.convolutional_nn(X.shape, n_classes=2)

#Normalisation. This is needed.
#FFNN's seem to throw a fit with un-normalised data.
X = gen_tools.normalise(X)

#nn.fit(X, Y, epochs=10, batch_size=45)

#Cross Validation
X, Y = shuffle(X, Y, random_state=0)
keras_classifiers.perform_crossval(X, Y, nn, n_splits=5, epochs=50, batch_size=20, opt=opt)

#standard_classifiers.csp_knn(raw, -1., 4., max_n_neighbors=8, n_jobs=-1, max_csp_components=10)
#For Subjects 1-10 Combined.
#CVGridSearch completed in 41021.308s
# Displaying Results...
# Best score: 0.542
# Best parameters set:
# 	CSP__cov_est: 'concat'
# 	CSP__n_components: 9
# 	CSP__norm_trace: True
# 	KNN__algorithm: 'ball_tree'
# 	KNN__n_neighbors: 7
# 	KNN__weights: 'uniform'
#
# Process finished with exit code 0
