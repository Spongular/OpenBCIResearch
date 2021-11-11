import mne
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler

import gen_tools
import keras_classifiers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import backend
import LiveBCI

#Gets around chance level accuracy. Likely not enough data for training and validation.
def EEGNet_test(epochs, labels):
    # Scale the data
    data = epochs.get_data() * 1000

    # Shuffle and Normalise our data
    data, labels = shuffle(data, labels, random_state=1)

    # Reshape the data and then expand our dimensions to fit the model.
    print("Data Set Before Reshape: {}".format(data.shape))
    print("Reshaping Data...")
    # data = np.reshape(data, (data.shape[0], data.shape[2], data.shape[1]))
    data = gen_tools.reshape_3to4(data)
    print("Data Set After Reshape: {}".format(data.shape))

    # Set the class weights (may need to look at this in the future).
    class_weights = {0: 1, 1: 1}

    # Split into train, test and val.
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.4, random_state=1)
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)

    # Generate model, optimiser and checkpointer.
    dropout = 0.2
    model, opt = keras_classifiers.convEEGNet(input_shape=X_train.shape, chan=4, n_classes=2, d_rate=dropout,
                                              first_tf_size=128)
    # model, opt = keras_classifiers.test(nb_classes=2, Chans=data.shape[1], Samples=data.shape[2], ThirdAxis=data.shape[3],
    #                                     dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,)
    filepath = "NN_Weights/convEEGNet/4-channel-headband/4-Channel-HeadbandLayout-OpenBCIData-MotorMovement--20%-dropout-{epoch:02d}-{val_accuracy:.2f}.h5"
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                   save_best_only=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    # Form the learning rate scheduler.
    scheduler = LearningRateScheduler(keras_classifiers.EEGNetScheduler)

    fittedModel = model.fit(X_train, y_train, batch_size=10, epochs=100,
                            verbose=2, validation_data=(X_val, y_val),
                            callbacks=[checkpointer, scheduler], class_weight=class_weights)
    return



#Clear the last session just in case.
backend.clear_session()

#Set the log level to warning to avoid flooding the output window.
mne.set_log_level('WARNING')

#Now we load the recordings
files = ['E:\\PycharmProjects\\OpenBCIResearch\\DataGathering\\LiveRecordings\\MotorResponses\\Movement\\subject1-10_06_2021-mm-lr-001_raw.fif',
         'E:\\PycharmProjects\\OpenBCIResearch\\DataGathering\\LiveRecordings\\MotorResponses\\Movement\\subject1-10_06_2021-mm-lr-002_raw.fif',
         'E:\\PycharmProjects\\OpenBCIResearch\\DataGathering\\LiveRecordings\\MotorResponses\\Movement\\subject1-10_06_2021-mm-lr-003_raw.fif',
         'E:\\PycharmProjects\\OpenBCIResearch\\DataGathering\\LiveRecordings\\MotorResponses\\Movement\\subject1-10_06_2021-mm-lr-004_raw.fif']

board = LiveBCI.generateOpenBCIBoard('COM3', 'ganglion', timeout=15)
test = LiveBCI.MotorImageryStimulator(stim_time=4, wait_time=4, stim_count=5, stim_type='lr', board=board)
test.load_multiple_data(files=files)

#The channel names weren't properly recorded for these datasets, so let's fix that.
ch_dict = {'CH1':'Fp1', 'CH2':'Fp2', 'CH3':'O1', 'CH3':'O2'}
test.raw.rename_channels(ch_dict)

#Then, we filter, epoch and resample to make the data match.
test.filter_raw(min=4.)
test.eeg_to_epochs(tmin=0., tmax=3., event_dict=dict(T1=2, T2=3), stim_ch='STI001')
#test.resample_epochs(new_rate=160)

epochs = test.epochs

#Grab our labels.
print(epochs.event_id)
labels = epochs.events[:, -1] - 2

#Perform the EEGNet test.
EEGNet_test(epochs, labels)

#Re-filter and epoch the data for non-nn tests
test.eeg_to_raw()
test.filter_raw(min=7., max=35.)
test.eeg_to_epochs(tmin=0., tmax=4., event_dict=dict(T1=2, T2=3), stim_ch='STI001')


