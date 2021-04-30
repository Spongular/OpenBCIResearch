
import mne
from mne.io import read_raw_edf
from tensorflow.python.keras.callbacks import ModelCheckpoint

import standard_classifiers
import data_loading
import gen_tools
import keras_classifiers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
from keras import models
from mne.datasets.eegbci import eegbci

mne.set_log_level('WARNING')

#load our data.
raw = data_loading.get_all_mi_between(1, 81, 2, ["088", "092", "100"])
raw = gen_tools.preprocess_highpass(raw, min=4., fir_design='firwin')


#classify it.
#standard_classifiers.csp_lda(raw, -1., 4.)
#standard_classifiers.csp_svm(raw, -1., 4.)
#standard_classifiers.csp_lda(raw, -1., 4., ["C3", "Cz", "C4"])

#standard_classifiers.pca_lda(raw, -1., 4., ["C3", "Cz", "C4"], n_jobs=2, pca_n_components=32)
#standard_classifiers.pca_svm(raw, -1., 4., ["C3", "Cz", "C4"], n_logspace=2, n_jobs=2, pca_n_components=32)
#standard_classifiers.pca_knn(raw, -1., 4., ["C3", "Cz", "C4"], max_n_neighbors=10, n_jobs=2, pca_n_components=32)

#Get Epochs
data, labels, epochs = gen_tools.epoch_data(raw, tmin=0, tmax=4, pick_list=[], plot_bads=True, eeg_reject_uV=600, scale=1000)
#del data, labels #We don't need these.

#Generate Wavelets from Epochs.
#X, Y = gen_tools.wavelet_transform_morlet(epochs, event_names=epochs.event_id,
                                          #f_low=6., f_high=30., )
#data = np.moveaxis(data, 1, 2)
data = np.expand_dims(data, 3)
#Reshape X for FF network
#X = gen_tools.reshape_4to3(X)

#Form the Neural Network.
#nn, opt = keras_classifiers.feedforward_nn(X.shape, n_classes=2, model_shape=[16, 8, 4, 8, 16], d_rate=0.5, b_norm=False)
#nn, opt = keras_classifiers.convolutional_nn(data.shape, n_classes=2)
#nn = keras_classifiers.EEGNet(2, samples=data.shape[2])

model = keras_classifiers.EEGNet(nb_classes=2, Chans=data.shape[1], Samples=data.shape[2],
               dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
               dropoutType='Dropout')

checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                               save_best_only=True)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

data, labels = shuffle(data, labels, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)
y_test = to_categorical(y_test, 2)

class_weights = {0:1, 1:1}

fittedModel = model.fit(X_train, y_train, batch_size = 16, epochs = 300,
                        verbose = 2, validation_data=(X_val, y_val),
                        class_weight=class_weights, callbacks=[checkpointer])

probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)
acc         = np.mean(preds == y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

#Normalisation. This is needed.
#FFNN's seem to throw a fit with un-normalised data.
#X = gen_tools.normalise(X)

#nn.fit(data, to_categorical(labels, 2), epochs=100, batch_size=64)

#Cross Validation
#X, Y = shuffle(X, Y, random_state=0)
#keras_classifiers.perform_crossval(data, labels, nn, n_splits=5, epochs=100, batch_size=64, opt=opt)

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
