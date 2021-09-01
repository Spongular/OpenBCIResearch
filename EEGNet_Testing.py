#This is an implementation of the EEGNet architecture described in the work:
#An Accurate EEGNet-based Motor-Imagery Brain–Computer Interface for Low-Power Edge Computing
#Available at: https://arxiv.org/abs/2004.00077

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

#Clear the last session just in case.
backend.clear_session()

#Set the log level to warning to avoid flooding the output window.
mne.set_log_level('WARNING')

#load our data.
raw = data_loading.get_all_mi_between(1, 50, 1, ["38", "088", "89", "092", "100", "104"])

#Run a Highpass filter to get rid of low-frequency drift and other issues.
#This isn't specified in the paper, but is common practice and thus assumed.
#raw = gen_tools.preprocess_highpass(raw, min=4., fir_design='firwin')

raw = gen_tools.preprocess_bandpass(raw, min=4, max=50)

#Epoch our data.
tmin = 0.
tmax = 4.
#(C3, C4 and Cz got 70% or so.)
#data, labels, epochs = gen_tools.epoch_data(raw, tmin=tmin, tmax=tmax, pick_list=["Fp1", "Fp2", "O1", "O2"], scale=1000)
data, labels, epochs = gen_tools.epoch_data(raw, tmin=tmin, tmax=tmax, scale=1000)
print(epochs.ch_names)
#We don't need the epoch objects as we have the data/labels.
del epochs

#Shuffle and Normalise our data
data, labels = shuffle(data, labels, random_state=1)

#Slice and dice, my brothers.
data, labels = gen_tools.slice_data(data, labels, 8)

#Reshape the data and then expand our dimensions to fit the model.
print("Data Set Before Reshape: {}".format(data.shape))
print("Reshaping Data...")
#data = np.reshape(data, (data.shape[0], data.shape[2], data.shape[1]))
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
#model, opt = keras_classifiers.convEEGNet(input_shape=X_train.shape, chan=4, n_classes=2, d_rate=dropout, first_tf_size=128)
#model, opt = keras_classifiers.test(nb_classes=2, Chans=data.shape[1], Samples=data.shape[2], ThirdAxis=data.shape[3],
#                                     dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,)
model, opt = keras_classifiers.fusionEEGNet(n_classes=2, chans=X_train.shape[1], samples=X_train.shape[2],
                                            third_axis=X_train.shape[3], l_rate=1E-4)
filepath = "NN_Weights/convEEGNet/4-channel-headband/4-Channel-HeadbandLayout-MotorMovement--50%-dropout-{epoch:02d}-{val_accuracy:.2f}.h5"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                               save_best_only=True)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

#Form the learning rate scheduler.
scheduler = LearningRateScheduler(keras_classifiers.EEGNetScheduler)

fittedModel = model.fit(X_train, y_train, epochs = 100,
                        verbose = 2, validation_data=(X_val, y_val),
                        callbacks=[checkpointer], class_weight=class_weights)

#fittedModel = model.fit(X_train, y_train, batch_size = 16, epochs = 100,
#                        verbose = 2, validation_data=(X_val, y_val),
#                        callbacks=[checkpointer, scheduler], class_weight=class_weights)

probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)
acc         = np.mean(preds == y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))





