#This is an implementation of the EEGNet architecture described in the work:
#Fusion Convolutional Neural Network for Cross-Subject EEG Motor Imagery Classification
#Available at: https://www.mdpi.com/2073-431X/9/3/72/pdf
#https://github.com/rootskar/EEGMotorImagery/blob/master/EEGModels.py

import mne
from mne.io import read_raw_edf
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler

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
from keras import backend
from matplotlib import pyplot as plt

#Clear the last session just in case.
backend.clear_session()

#Set the log level to warning to avoid flooding the output window.
mne.set_log_level('WARNING')

#load our data.
raw = data_loading.get_all_mi_between(1, 110, 1, ["38", "088", "89", "092", "100", "104"])

#preprocess
raw = gen_tools.preprocess_bandpass(raw, min=4, max=50)

#Epoch our data.
tmin = 0.
tmax = 4.
data, labels, epochs = gen_tools.epoch_data(raw, tmin=tmin, tmax=tmax, scale=1000)
print(epochs.ch_names)
#We don't need the epoch objects as we have the data/labels.
del epochs

#Slice and dice, my brothers.
slice_count = 8
print("Data Set Before Slicing: {}".format(data.shape))
print("Slicing Data Into {count} Pieces...".format(count=slice_count))
data, labels = gen_tools.slice_data(data, labels, slice_count)
print("Data Set After Slicing: {}".format(data.shape))

#Shuffle and Normalise our data
data, labels = shuffle(data, labels, random_state=1)

#Reshape the data and then expand our dimensions to fit the model.
print("Data Set Before Reshape: {}".format(data.shape))
print("Reshaping Data...")
data = gen_tools.reshape_3to4(data)
print("Data Set After Reshape: {}".format(data.shape))

#Set the class weights (may need to look at this in the future).
class_weights = {0:1, 1:1}

#Split into train, test and val.
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)
y_test = to_categorical(y_test, 2)

#Generate model, optimiser and checkpointer.
dropout = 0.5
model, opt = keras_classifiers.fusionEEGNet(n_classes=2, chans=X_train.shape[1], samples=X_train.shape[2],
                                            third_axis=X_train.shape[3], l_rate=1E-4, dropout_rate=dropout)
filepath = "NN_Weights/convEEGNet/4-channel-headband/4-Channel-HeadbandLayout-MotorMovement--50%-dropout-{epoch:02d}-{val_accuracy:.2f}.h5"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                               save_best_only=True)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

#Form the learning rate scheduler.
scheduler = LearningRateScheduler(keras_classifiers.EEGNetScheduler)

fittedModel = model.fit(X_train, y_train, epochs = 100,
                        verbose = 2, validation_data=(X_val, y_val),
                        class_weight=class_weights)
#Add callbacks=[checkpointer, scheduler] to include callbacks

probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)
acc         = np.mean(preds == y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

#Plot Accuracy and Loss
print(fittedModel.history.keys())

#Acc
plt.plot(fittedModel.history['accuracy'])
plt.plot(fittedModel.history['val_accuracy'])
plt.title('{name} - Accuracy'.format(name=model.name))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show(block=True)

#Loss
plt.plot(fittedModel.history['loss'])
plt.plot(fittedModel.history['val_loss'])
plt.title('{name} - Loss'.format(name=model.name))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show(block=True)