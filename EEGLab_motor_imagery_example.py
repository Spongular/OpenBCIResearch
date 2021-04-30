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
#Apparently, scaling the data by 1000 helps classification.
data, labels, epochs = gen_tools.epoch_data(raw, tmin=0, tmax=4, pick_list=[], plot_bads=True, eeg_reject_uV=600, scale=1000)

#An extra dimension is needed to fit expected input for Conv2d layer in EEGNet.
data = np.expand_dims(data, 3)

#Form the Neural Network.
model = keras_classifiers.EEGNet(nb_classes=2, Chans=data.shape[1], Samples=data.shape[2],
               dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
               dropoutType='Dropout')

#This allows us to keep track of the set of node weights that offer the best result.
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                               save_best_only=True)

#Compile the model with categorical cross-entropy.
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

#Shuffle the data, and split into train/test/val.
data, labels = shuffle(data, labels, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

#Change the labels to categorical to fit with expected input for categorical cross-entropy.
y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)
y_test = to_categorical(y_test, 2)

#Set the class weights to one each.
class_weights = {0:1, 1:1}

#Fit the model for 300 epochs. This may be excessive, but provides plenty of time to find the best
#fit for the data. Using validation_split seems to not agree with the CNN, so don't.
fittedModel = model.fit(X_train, y_train, batch_size = 16, epochs = 300,
                        verbose = 2, validation_data=(X_val, y_val),
                        class_weight=class_weights, callbacks=[checkpointer])

#Predict on the test set and find our final accuracy on test data.
probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)
acc         = np.mean(preds == y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

#First run gave:

# Epoch 00300: val_loss did not improve from 0.39680
# Classification accuracy: 0.793160
#
# Process finished with exit code 0