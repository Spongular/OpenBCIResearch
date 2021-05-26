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
from keras import models, optimizers
from mne.datasets.eegbci import eegbci
from keras import backend

def runPremadeEEGNet(raw, highpass=4., tmin=0., tmax=4., batch_size=16, epoch_count=300,
                     learn_rate=None, scheduler=None, checkpoint=None):
    #Filter and Epoch
    raw = gen_tools.preprocess_highpass(raw, min=highpass, fir_design='firwin')
    data, labels, epochs = gen_tools.epoch_data(raw, tmin=tmin, tmax=tmax, pick_list=[], plot_bads=False, eeg_reject_uV=None,
                                                scale=1000)
    print("T1: %f" % (len(epochs['T1'])))
    print("T1: %f" % (len(epochs['T2'])))

    #Expand the dimensions and split into train/validation sets
    data = np.expand_dims(data, 3)
    data, labels = shuffle(data, labels, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    y_test = to_categorical(y_test, 2)

    #Generate EEGNet Model
    model = keras_classifiers.EEGNet(nb_classes=2, Chans=data.shape[1], Samples=data.shape[2], ThirdAxis=data.shape[3],
                                     dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
                                     dropoutType='Dropout')

    #Here we set up our callbacks. They are used by the fit procedure to perform tasks during fitting,
    #such as saving checkpoints when validation loss decreases, or changing learning rates.
    calls = []
    if checkpoint is None:
        checkpoint = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                       save_best_only=True)
        calls.append(checkpoint)
    else:
        calls.append(checkpoint)
    if scheduler is not None:
        calls.append(scheduler)

    opt = optimizers.Adam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    #Form class weights and fit the model to the data
    class_weights = {0: 1, 1: 1}

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch_count,
              verbose=2, validation_data=(X_val, y_val),
              class_weight=class_weights, callbacks=calls)

    #Print test values
    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))

    #Return the model for further use.
    #This method reaches around ~80% accuracy with all 64 channels.
    return model

def runConvEEGNet(raw, highpass=4., tmin=0., tmax=4., batch_size=16, epoch_count=300,
                     learn_rate=None, scheduler=None, checkpoint=None):
    raw = gen_tools.preprocess_highpass(raw, min=highpass, fir_design='firwin')
    data, labels, epochs = gen_tools.epoch_data(raw, tmin=tmin, tmax=tmax, pick_list=[], plot_bads=False,
                                                eeg_reject_uV=None,
                                                scale=1000)
    print("T1: %f" % (len(epochs['T1'])))
    print("T1: %f" % (len(epochs['T2'])))

    # Expand the dimensions and split into train/validation sets
    data = np.expand_dims(data, 3)
    data, labels = shuffle(data, labels, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    y_test = to_categorical(y_test, 2)

    calls = []
    if checkpoint is None:
        checkpoint = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                     save_best_only=True)
        calls.append(checkpoint)
    else:
        calls.append(checkpoint)
    if scheduler is not None:
        calls.append(scheduler)
    else:
        # Form the learning rate scheduler.
        scheduler = LearningRateScheduler(keras_classifiers.EEGNetScheduler)
        calls.append(scheduler)

    # Generate model, optimiser and checkpointer.
    dropout = 0.0
    model, opt = keras_classifiers.convEEGNet(input_shape=X_train.shape, chan=3, n_classes=2, d_rate=dropout,
                                              first_tf_size=128)

    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    # Form class weights and fit the model to the data
    class_weights = {0: 1, 1: 1}

    fittedModel = model.fit(X_train, y_train, batch_size=16, epochs=100,
                            verbose=2, validation_data=(X_val, y_val),
                            callbacks=calls, class_weight=class_weights)
    return model