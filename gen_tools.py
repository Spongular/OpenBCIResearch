#This file contains all of the pre-processing methods and generally useful tools for the project.

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import pick_types, Epochs, events_from_annotations, concatenate_raws
from mne.channels import make_standard_montage
from mne.decoding import CSP, Scaler, UnsupervisedSpatialFilter, Vectorizer
from mne.datasets.eegbci import eegbci
from mne.time_frequency import psd_welch, psd_multitaper, tfr_morlet
from keras.utils import to_categorical
from mne.io import read_raw_edf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split, GridSearchCV, HalvingGridSearchCV, \
    #RandomizedSearchCV, HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from time import time

#General miscellaneous settings for raw MI from physionet plus a bandpass.
def mifixes_and_bandpass(raw, min=1., max=40., fir_design='firwin'):
    print("Standardising Raw, setting montage and fixing channel names...")
    #Fix our settings for the raw.
    eegbci.standardize(raw)
    raw.set_montage("standard_1020", match_case=False)
    raw.rename_channels(lambda s: s.strip("."))
    # Filter it.
    print("Performing bandpass filter in range %f to %f" % (min, max))
    raw.filter(min, max, fir_design=fir_design, skip_by_annotation='edge')  # Bandpass
    return raw

#Epochs the raw data and returns what we want.
def epoch_data(raw, tmin, tmax, pick_list=[]):
    print("Extracting epochs from raw...")
    # First, grab the events.
    events, event_id = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    # We pick our channels by type and name and grab epochs.
    if len(pick_list) > 0:
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads',
                           selection=pick_list)
    else:
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1] - 2
    data = epochs.get_data()
    return data, labels, epochs

#Performs a wavelet transform on the epochs passed.
#Heavily inspired by: https://github.com/kylemath/DeepEEG/blob/7e2461bb26975589c529110d691a41b6d184ca58/utils.py#L504
#Returns an X and Y set for use in machine learning.
def wavelet_transform(epochs, event_names=[], f_low=1., f_high=40., f_num=6,
                      wave_cycles=3, wavelet_decim=1, picks=[], baseline=[-1,-0.5]):
    tfr_dict = {} #This will contain our wavelet transforms.
    frequencies = np.linspace(f_low, f_high, f_num, endpoint=True)
    if len(event_names) < 1: #If our event names are empty, just use all of them.
        event_names = list(epochs.event_id)
    if len(picks) < 1: #If our picks are empty, grab them all.
        picks = epochs.ch_names
    #Now, for each event type, calculate the set of wavelet transforms per event.
    for event in event_names:
        print('Performing Morlet Wavelet Transform on ' + event)
        #For the morlet function, return_itc and average both need to be false,
        #or else the function will fail, or return a single average of all epochs.
        tfr_temp = tfr_morlet(epochs[event], freqs=frequencies, n_cycles=wave_cycles,
                              return_itc=False, picks=picks, average=False,
                              decim=wavelet_decim, output='power')
        #Most instances of TFR involve applying a baseline based on the first 0.5-1 second
        #of the epoch. Nothing I've found can explain why.
        tfr_temp = tfr_temp.apply_baseline(baseline, mode='mean')

        #Now, reshaping? There is no explanation as to why...
        #From what I can see, it seems to change the order from: [epoch, channels, wavelets, time]
        #to the order: [epoch, time, wavelet, channel]. I'm not quite sure why, but it does make sense.
        #Seems to be in line with the format for image classification, i.e. [pixel_x, pixel_y, rgb_channels]
        #Ultimately, it shouldn't really make much difference...
        #The DeepEEG method seems to cut the data to stimulation onset and onward.
        #This isn't necessarily what we want, as ERS/ERD occurs both before, during and after stim.
        #This can be seen in our wavelet diagrams quite clearly.
        #stim_onset = np.argmax(tfr_temp.times > 0)
        #power_out_temp = np.moveaxis(tfr_temp.data[:,:,:,stim_onset:], 1, 3)
        power_out_temp = np.moveaxis(tfr_temp.data[:, :, :, :], 1, 3)
        power_out_temp = np.moveaxis(power_out_temp, 1, 2)
        tfr_dict[event] = power_out_temp
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
    return X, Y

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

def normalise(data):
    norm_data = (data - np.mean(data)) / np.std(data)
    norm_data = norm_data.astype('float32')
    return norm_data
