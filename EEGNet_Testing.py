#This is an implementation of the EEGNet architecture described in the work:
#An Accurate EEGNet-based Motor-Imagery Brain–Computer Interface for Low-Power Edge Computing
#Available at: https://arxiv.org/abs/2004.00077

import mne
from mne.io import read_raw_edf
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping

import standard_classifiers
import os
import data_loading
import gen_tools
import keras_classifiers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras import models
from mne.datasets.eegbci import eegbci
from keras import backend
from matplotlib import pyplot as plt
import random
from sklearn.metrics import accuracy_score

#Clear the last session just in case.
backend.clear_session()

#Set the log level to warning to avoid flooding the output window.
mne.set_log_level('WARNING')

#Time periods for the epochs
tmin = 0.
tmax = 4.

"""Parameters:
    -EEGNet:
        Dropout = 0%. 25% and 50% result in 65% accuracy and 75% accuracy, but 0% gives 80%+
        LR Scheduler = 0.01, 0.001, 0.0001 at 0-20, 20-50 and 50-100 respectively.
    -EEGNetFusion:
        Dropout = 50%, LR = 1E-4, No Schedule = 0.773064 Avg (64ch), 0.633210 Avg (Headband), 0.593989 Avg (Cortex)
        Dropout = 0%, LR Scheduler, Earlystop = 0.821029 Avg (64ch), 0.657270 Avg (Headband), 0.598130 Avg (Cortex)
        Dropout = 0%, lr = 1E-4, No Earlystop = 0.612737 Avg (Headband)
        Same as Above with batch size 16 = 0.625835
    -ShallowConvNet:
        0.25 dropout 0.783445 Avg
        
        
        
"""
classifier = 'deep-convnet'
pick_lists = [[], ["Fp1", "Fp2", "O1", "O2"], ["C3", "Cz", "C4"]]
picks = pick_lists[0]

#load our data.
subjects = set(range(1, 110))
exclusions = set([38, 80, 88, 89, 92, 100, 104])
subjects = subjects - exclusions
sub_dict = {}
for sub in subjects:
    raw = data_loading.get_single_mi(sub, 2)
    raw.notch_filter(60, filter_length='auto', phase='zero')
    raw = gen_tools.preprocess_bandpass(raw, min=2., max=60.)
    #raw = gen_tools.preprocess_highpass(raw, min=4., fir_design='firwin')
    data, labels, epochs = gen_tools.epoch_data(raw, tmin=tmin, tmax=tmax, scale=1000, pick_list=picks)
    del epochs
    #data, labels = gen_tools.slice_data(data, labels, slice=8)
    sub_dict[sub] = (data, labels)

test_size = 20
test_set = set([])
tested_set = set([])
sub_set = set(list(sub_dict.keys()))
res = []
res2 = []
for x in range(1, 6):
    #Form the test and train sets.
    if (len(sub_set) - len(tested_set)) < test_size:
        tested_set = set([])
    test_set = set(random.sample((sub_set - tested_set), test_size))
    train_set = sub_set - test_set
    tested_set = tested_set.union(test_set)

    #Print to keep track.
    print("Iteration: {n}".format(n=x))
    print("\nTest Set: {t}\n".format(t=test_set))
    print("Train Set: {t}\n".format(t=train_set))
    print("Tested Set: {t}\n".format(t=tested_set))

    #Concatenate data
    t = test_set.pop()
    X_test = sub_dict[t][0]
    Y_test = sub_dict[t][1]
    while len(test_set) > 0:
        t = test_set.pop()
        X_test = np.concatenate((X_test, sub_dict[t][0]), axis=0)
        Y_test = np.concatenate((Y_test, sub_dict[t][1]), axis=0)

    # Concat, shuffle and form val set.
    t = train_set.pop()
    X_train = sub_dict[t][0]
    Y_train = sub_dict[t][1]
    while len(train_set) > 0:
        t = train_set.pop()
        X_train = np.concatenate((X_train, sub_dict[t][0]), axis=0)
        Y_train = np.concatenate((Y_train, sub_dict[t][1]), axis=0)

    #Reshape, split, and one-hot

    X_test = gen_tools.reshape_3to4(X_test)
    X_train = gen_tools.reshape_3to4(X_train)
    print(X_test.shape)
    print(X_train.shape)
    #X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1], X_test.shape[3])
    #X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1], X_train.shape[3])
    #print(X_test.shape)
    #print(X_train.shape)
    X_train, Y_train = shuffle(X_train, Y_train)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.125)
    Y_train = to_categorical(Y_train, 2)
    Y_val = to_categorical(Y_val, 2)
    Y_test = to_categorical(Y_test, 2)
    # Set the class weights (may need to look at this in the future).
    class_weights = {0: 1, 1: 1}

    X_test, Y_test = shuffle(X_test, Y_test)

    #Form classifier
    # Generate model, optimiser and checkpointer.
    if classifier == 'eegnet':
        dropout = 0.5
        model, opt = keras_classifiers.convEEGNet(input_shape=X_train.shape, chan=data.shape[1], n_classes=2,
                                                  d_rate=dropout, first_tf_size=128,
                                                  l_rate=0.0001)
        filepath = "NN_Weights/eegnet_best.hdf5"
        if os.path.isfile(filepath):
            os.remove(filepath)
            print("File {file} removed.".format(file=filepath))

        #  Make our callbacks
        scheduler = LearningRateScheduler(keras_classifiers.EEGNetScheduler)
        #earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True)
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                       save_best_only=True, save_weights_only=True)

        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])

        fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=100,
                                verbose=2, validation_data=(X_val, Y_val),
                                callbacks=[checkpointer, scheduler], class_weight=class_weights)

        model.load_weights(filepath)


    elif classifier == 'eegnet-fusion':
        dropout = 0.5
        model, opt = keras_classifiers.fusionEEGNet(n_classes=2, chans=X_train.shape[1], samples=X_train.shape[2],
                                                    third_axis=X_train.shape[3], l_rate=0.001, dropout_rate=dropout)

        filepath = "NN_Weights/eegnet_fusion_best.hdf5"
        if os.path.isfile(filepath):
            os.remove(filepath)
            print("File {file} removed.".format(file=filepath))

        #  Make our callbacks
        #scheduler = LearningRateScheduler(keras_classifiers.EEGNetScheduler)
        #earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True)
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                       save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])

        #Use default batch size.
        fittedModel = model.fit(X_train, Y_train, batch_size=64, epochs=100,
                                verbose=2, validation_data=(X_val, Y_val),
                                callbacks=[checkpointer,reduce_lr], class_weight=class_weights)
        model.load_weights(filepath)

    elif classifier == 'shallow-convnet':
        dropout = 0.5
        model, opt = keras_classifiers.ShallowConvNet(nb_classes=2, Chans=X_train.shape[1], Samples=X_train.shape[2],
                                                      l_rate=1E-4, dropoutRate=dropout)
        filepath = "NN_Weights/shallowconv_best.hdf5"
        if os.path.isfile(filepath):
            os.remove(filepath)
            print("File {file} removed.".format(file=filepath))

        #  Make our callbacks
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                       save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])

        # Use default batch size.
        fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=100,
                                verbose=2, validation_data=(X_val, Y_val),
                                callbacks=[checkpointer], class_weight=class_weights)
        model.load_weights(filepath)

    elif classifier == 'deep-convnet':
        dropout = 0.5
        model, opt = keras_classifiers.DeepConvNet(nb_classes=2, Chans=X_train.shape[1], Samples=X_train.shape[2],
                                                   l_rate=0.001, dropoutRate=dropout)
        filepath = "NN_Weights/deepconv_best.hdf5"
        if os.path.isfile(filepath):
            os.remove(filepath)
            print("File {file} removed.".format(file=filepath))

        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                       save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])

        # Use default batch size.
        fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=100,
                                verbose=2, validation_data=(X_val, Y_val),
                                callbacks=[checkpointer, reduce_lr], class_weight=class_weights)
        model.load_weights(filepath)


    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))
    res.append(acc)
    print("Classification accuracy (Normal Method): %f " % (acc))

    labs = Y_test.argmax(axis=-1)
    acc = accuracy_score(labs, preds)
    res2.append(acc)
    print("Classification accuracy (Sklearn Metric): {a}".format(a=acc))

    #Fix the tested set
    tested_set = tested_set.union(test_set)

print(res)
print("Average Accuracy (Normal Method) = {ave}".format(ave=np.mean(res)))
print(res2)
print("Average Accuracy (Sklearn Metric) = {ave}".format(ave=np.mean(res2)))

# #Plot Accuracy and Loss
# print(fittedModel.history.keys())
#
# #Acc
# plt.plot(fittedModel.history['accuracy'])
# plt.plot(fittedModel.history['val_accuracy'])
# plt.title('{name} - Accuracy'.format(name=model.name))
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training', 'Validation'], loc='upper left')
# plt.show(block=True)
#
# #Loss
# plt.plot(fittedModel.history['loss'])
# plt.plot(fittedModel.history['val_loss'])
# plt.title('{name} - Loss'.format(name=model.name))
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show(block=True)


