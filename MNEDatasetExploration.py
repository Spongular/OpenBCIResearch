#Source: https://mne.tools/dev/auto_examples/decoding/plot_decoding_csp_eeg.html

#Our imports. I'm being lazy and just importing the whole lot.

#Here, we can set our details for the epoch gathering.
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import concatenate_raws, events_from_annotations, pick_types, Epochs
from mne.channels import make_standard_montage
from mne.datasets.eegbci import eegbci
from mne.decoding import CSP
from mne.io import read_raw_edf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

#Filtering.
#First, we band-pass, then split into epochs, then DSP, Average and DSP again.
#DSP1 = x*x, DSP2 = log(1+x)
#def filterEpochs(ep1, ep2, )

#This function concatenates EDF files and returns them.
#It also sets basic parameters about the raws.
def concatRaws(files):
    #load the files
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
    #standardise them.
    eegbci.standardize(raw)
    #This is mostly for visualisations
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage)
    #This removes that annoying dot.
    raw.rename_channels(lambda x: x.strip('.'))
    return raw

#This is used to get a set of epochs.
#The data itself is only ran through a bandpass filter.
def getEpochs(raw, tmin, tmax, filter_l, filter_h, selected_channels):
    # Apply a band-pass filter.
    # In this instance, the lower bound is 7hz, and upper is 30hz.
    # The skip_by_annotation setting catches any leftovers from the concat.
    raw.filter(filter_l, filter_h, fir_design='firwin', skip_by_annotation='edge')
    # Now we grab events and print them.
    events, event_id = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    # Next, we pick our channels by type and name.
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, selection=selected_channels,
                       exclude='bads')
    # Now we can finally get our epochs. Training will be done between 1s and 2s in the epochs.
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    return epochs

def getEpochs(raw, tmin, tmax, filter_l, filter_h):
    # Apply a band-pass filter.
    # In this instance, the lower bound is 7hz, and upper is 30hz.
    # The skip_by_annotation setting catches any leftovers from the concat.
    raw.filter(filter_l, filter_h, fir_design='firwin', skip_by_annotation='edge')
    # Now we grab events and print them.
    events, event_id = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    # Next, we pick our channels by type and name.
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    # Now we can finally get our epochs. Training will be done between 1s and 2s in the epochs.
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    return epochs

def plotEvoked(epochs):
    evoked = epochs.average()
    evoked.plot()
    plt.show(block=True)

def cleanData(epochs):
    epochs.equalize_epoch_counts(['T1', 'T2'])
    return epochs

#Function for testing with LDA
#tmin and tmax are the times around the event for epochs.
#e.g. tmin = -1 means epochs start a second before an event.
def concatAndTest(files, tmin, tmax):
    print(files)
    raw = concatRaws(files)
    epochs = getEpochs(raw, -1, 4, 7., 30.)
    epochs_train = epochs.copy()
    labels = epochs.events[:,-1] -2
    # Now for a monte-carlo cross-validation generator
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)
    #Assemble classifiers
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    #Use scikit-learn pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
    #And print them
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                              class_balance))
    return np.mean(scores)

# MI Dataset, LDA with 5-Fold Cross-Validation, No cross-subject training/testing
def runTest1():
    sumAccuracy = 0
    sumGood = 0
    failCases = list()
    runCount = 0
    for a in range(1, 87):
        num = "{0:03}".format(a)
        files = ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R06.edf",
                 "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R10.edf",
                 "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R14.edf"]
        print("Test Set 00" + str(a) + ":")
        acc = concatAndTest(files, -1, 4)
        sumAccuracy += acc
        if acc > 0.75:
            sumGood += 1
        else:
            failCases.append(a)
        print("\n")
        runCount += 1
    avgAcc = sumAccuracy / runCount
    print("Overall Average Accuracy: " + str(avgAcc))
    print("Ratio of > 75% classification rates: " + str(sumGood / runCount))
    print("Test Sets with Accuracy < 75%:")
    print(failCases)

# MI Dataset, LDA with 5-Fold Cross-Validation, Mixed Subjects (Grouping in 3s).
#Subjects 1-87
def runTest2():
    sumAccuracy = 0
    sumGood = 0
    failCases = list()
    runCount = 0
    for a  in range(1, 84, 3):
        num = "{0:03}".format(a)
        files = ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R06.edf",
                 "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R10.edf",
                 "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R14.edf"]
        num = "{0:03}".format(a + 1)
        files += ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R06.edf",
                 "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R10.edf",
                 "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R14.edf"]
        num = "{0:03}".format(a + 2)
        files += ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R06.edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R10.edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R14.edf"]

        print("Test Sets " + str(a) + ", " + str(a + 1) + ", " + str(a+2) + ":")
        acc = concatAndTest(files, -1, 4, ['C3', 'C4', 'Cz'])
        sumAccuracy += acc
        if acc > 0.75:
            sumGood += 1
        else:
            failCases.append(a)
        print("\n")
        runCount += 1
    avgAcc = sumAccuracy / runCount
    print("Overall Average Accuracy: " + str(avgAcc))
    print("Ratio of > 75% classification rates: " + str(sumGood / runCount))
    print("Test Sets with Accuracy < 75%:")
    print(failCases)

def runTest3():
    print("Open and close left and right hands:")
    files = []
    for a in range(1, 80):
        num = "{0:03}".format(a)
        files += ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R03.edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R07.edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R11.edf"]
    concatAndTest(files, -1, 4, ['C3', 'C4', 'Cz'])
    print("\n\n")

    print("Imagine open and close left and right hands:")
    files = []
    for a in range(1, 80):
        num = "{0:03}".format(a)
        files += ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R04.edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R08.edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R12.edf"]
    concatAndTest(files, -1, 4, ['C3', 'C4', 'Cz'])
    print("\n\n")

    print("Open and close both hands and both feet:")
    files = []
    for a in range(1, 80):
        num = "{0:03}".format(a)
        files += ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R05.edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R09.edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R13.edf"]
    concatAndTest(files, -1, 4, ['C3', 'C4', 'Cz'])
    print("\n\n")

    print("Imagine open and close both hands and both feet:")
    files = []
    for a in range(1, 80):
        num = "{0:03}".format(a)
        files += ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R06.edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R10.edf",
                  "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R14.edf"]
    concatAndTest(files, -1, 4, ['C3', 'C4', 'Cz'])
    print("\n\n")

#Make sure that the log level is warning or we get a spew of shit.
mne.set_log_level('WARNING')
# a = 3
# num = "{0:03}".format(a)
# files = ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R06.edf",
#          "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R10.edf",
#          "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R14.edf"]
#
# #Use our functions to grab the files.
# raw = concatRaws(files)
# epochs = getEpochs(raw, -1, 4, 7., 30., ['C3', 'C4', 'Cz'])
#
# #Split our epochs between events.
# epoch_x = epochs['T1']
# epoch_y = epochs['T2']
#
# plotEvoked(epoch_x)
# plotEvoked(epoch_y)
# plt.show(block=True)
#
# concatAndTest(files, -1, 4, ['C3', 'C4', 'Cz'])

runTest1()