#This file contains all of the pre-processing methods and generally useful tools for the project.

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import pick_types, Epochs, events_from_annotations, concatenate_raws, find_events
from mne.channels import make_standard_montage
from mne.decoding import CSP, Scaler, UnsupervisedSpatialFilter, Vectorizer
from mne.datasets.eegbci import eegbci
from mne.time_frequency import psd_welch, psd_multitaper, tfr_morlet, tfr_multitaper, tfr_stockwell
from keras.utils import to_categorical
from mne.io import read_raw_edf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn import utils
from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split, GridSearchCV, HalvingGridSearchCV, \
    #RandomizedSearchCV, HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.utils import shuffle

#General miscellaneous settings for raw MI from physionet plus a bandpass.
def preprocess_bandpass(raw, min=1., max=40., fir_design='firwin'):
    print("Standardising Raw, setting montage and fixing channel names...")
    #Fix our settings for the raw.
    eegbci.standardize(raw)
    raw.set_montage("standard_1020", match_case=False)
    raw.rename_channels(lambda s: s.strip("."))
    # Filter it.
    print("Performing bandpass filter in range %f to %f" % (min, max))
    raw.filter(min, max, fir_design=fir_design, skip_by_annotation='edge')  # Bandpass
    return raw

def preprocess_highpass(raw, min=1., fir_design='firwin', notch_val=60):
    print("Standardising Raw, setting montage and fixing channel names...")
    #Fix our settings for the raw.
    eegbci.standardize(raw)
    raw.set_montage("standard_1020", match_case=False)
    raw.rename_channels(lambda s: s.strip("."))
    # Filter it.
    print("Performing Notch Filtering at {n} to remove line noise...".format(n=notch_val))
    raw.notch_filter(notch_val, filter_length='auto', phase='zero')
    print("Performing highpass filter in at %fHz..." % min)
    raw.filter(min, None, fir_design=fir_design, skip_by_annotation=('edge', 'bad_acq_skip'))  # Bandpass
    return raw

#Epochs the raw data and returns what we want.
def epoch_data(raw, tmin, tmax, pick_list=[], plot_bads=False, eeg_reject_uV=None, scale=None):
    print("Extracting epochs from raw...")
    # First, grab the events.
    events, event_id = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

    # We pick our channels by type and name and grab epochs.
    if len(pick_list) > 0:
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads',
                           selection=pick_list)
    else:
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    #Now, if we have no reject, we perform it normally, otherwise we drop epochs based on the threshold.
    if eeg_reject_uV is None:
        epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                        baseline=None, preload=True)
    else:
        epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True, reject=dict(eeg=eeg_reject_uV * 1e-6))

    epochs.drop_bad()
    print(epochs.drop_log)

    #Plot our bads.
    if plot_bads:
        epochs.plot_drop_log()

    #Grab our labels.
    print(epochs.event_id)
    labels = epochs.events[:, -1] - 2

    #We can scale the data here by a given value. Some NN implementations benefit from scaling the data
    #updwards by a few orders of magnitude, i.e. scale=1000.
    if scale is None:
        data = epochs.get_data()
    else:
        data = epochs.get_data() * scale

    return data, labels, epochs

def epoch_ssvep_MAMEM(raw, tmin, tmax, pick_list=[]):
    event_id = {"6.66": 1, "7.50": 2, "8.57": 3, "10.00": 4, "12.00": 5}
    events = find_events(raw=raw, stim_channel='stim')
    if len(pick_list) > 0:
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads',
                           selection=pick_list)
    else:
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    # Additionally, for our system, we want the 10-20 equivalents for the channels, not all 256, so...
    ssvep_epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                          baseline=None, preload=True, picks=picks)
    return ssvep_epochs

#Performs a Fast Fourier Transform on each epoch.
def fastfourier_transform(epochs):
    return

#Reshapes from 4 elements to 3.
#[epoch, time, wavelet, channel] to [epoch, time, wavelet*channel]
def reshape_4to3(data):
    reshaped_data = np.reshape(data, (data.shape[0], data.shape[1], np.prod(data.shape[2:])))
    return reshaped_data

#Reshapes from 4 elements to 2.
#[epoch, time, wavelet, channel] to [epoch, time*wavelet*channel]
def reshape_4to2(X):
    reshaped_data = np.reshape(X, (X.shape[0],np.prod(X.shape[1:])))
    return reshaped_data

#Expands from four elements to 5.
#[epoch, time, wavelet, channel] to [epoch, time, wavelet, channel, 1]
def reshape_4to5(data):
    reshaped_data = np.expand_dims(data, 4)
    return reshaped_data

def reshape_3to4(data):
    reshaped_data = np.expand_dims(data, 3)
    return reshaped_data

def normalise(data):
    norm_data = (data - np.mean(data)) / np.std(data)
    norm_data = norm_data.astype('float32')
    return norm_data

#inspired by: https://github.com/kylemath/DeepEEG/blob/7e2461bb26975589c529110d691a41b6d184ca58/utils.py#L504
def wavelet_transform_general(epochs, event_names=[], f_low=1., f_high=40., f_num=6,
                             wave_cycles=3, wavelet_decim=1, picks=[], baseline=[0,0.5],
                              transform_type='morlet', shuffle=True):
    #Create our dictionary and set of frequencies.
    tfr_dict = {}
    frequencies = np.linspace(f_low, f_high, f_num, endpoint=True)

    # Check the arguments.
    if len(event_names) < 1:
        event_names = list(epochs.event_id)
    if len(picks) < 1:
        picks = epochs.ch_names

    #Now, for each event type, calculate the set of wavelet transforms per event.
    for event in event_names:
        #For the tfr functions, return_itc and average must both be false.
        #In this case, we have three options: morlet, multitaper and stockwell.
        if transform_type == 'morlet':
            print('Creating Morlet Time Frequency Representation of ' + event)
            tfr_temp = tfr_morlet(epochs[event], freqs=frequencies, n_cycles=wave_cycles,
                                  return_itc=False, picks=picks, average=False,
                                  decim=wavelet_decim, output='power')
        elif transform_type == 'multitaper':
            print('Creating Multitaper Time Frequency Representation of ' + event)
            tfr_temp = tfr_multitaper(epochs[event], freqs=frequencies, n_cycles=wave_cycles,
                                  return_itc=False, picks=picks, average=False,
                                  decim=wavelet_decim)
        else:
            print('Creating Stockwell Time Frequency Representation of ' + event)
            tfr_temp = tfr_stockwell(epochs[event], freqs=frequencies, n_cycles=wave_cycles,
                                  return_itc=False, picks=picks, average=False,
                                  decim=wavelet_decim)

        #Most instances of TFR involve applying a baseline based on the first 0.5-1 second
        #of the epoch. Nothing I've found can explain why.
        tfr_temp = tfr_temp.apply_baseline(baseline, mode='mean')

        #No Reshaping here, just output it all bro.
        tfr_dict[event] = tfr_temp.data

    #Use the enumerate function to get a number for each event in event_names
    for ievent, event in enumerate(event_names):
        if ievent == 0:
            #For the first event, we fill Y with zeroes for each.
            X = tfr_dict[event]
            Y = np.zeros(len(tfr_dict[event]))
        else:
            #After the first, each event is given the number equal to ievent.
            X = np.append(X, tfr_dict[event], 0)
            Y = np.append(Y, np.ones(len(tfr_dict[event])) * ievent, 0)

    #Finally, shuffle X and Y before returning if shuffle is on.
    #Doing so ensures that there isn't any issues with ordering data for classification.
    if shuffle:
        X, Y = utils.shuffle(X, Y, random_state=1)
    return X, Y

import keras.backend as K

#From: https://stackoverflow.com/questions/43345909/when-using-mectrics-in-model-compile-in-keras-report-valueerror-unknown-metr/43354147#43354147
#Can be used in compile() as a metric.
def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def slice_data(data, labels):
    return data, labels