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
from sklearn import svm
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

mne.set_log_level('WARNING')
tmin, tmax = -1., 4. #This determines the boundaries for the epochs.
event_dict = dict(hands=2, feet=3) #Define our events.

#Select the hands & feet tests.
files = []
for a in range(1, 80):
    num = "{0:03}".format(a)
    files += ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R06.edf",
              "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R10.edf",
              "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R14.edf"]

#load the files.
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])

#This essentially ensures that the channel names meet the expected standard.
eegbci.standardize(raw)

#You're gonna need a montage.
montage = make_standard_montage('standard_1020')
raw.set_montage(montage)

#Stripe the "." character from channels.
raw.rename_channels(lambda x: x.strip('.'))

#Apply a band-pass filter.
#In this instance, the lower bound is 7hz, and upper is 30hz.
#The skip_by_annotation setting catches any leftovers from the concat.
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

#Now we grab events.
events, event_id = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

#Next, we pick our channels by type and name.
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

#Now we can finally get our epochs. Training will be done between 1s and 2s in the epochs.
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
epochs_train = epochs.copy().crop(tmin = 1., tmax = 2.)
labels = epochs.events[:,-1] -2

# Now for a monte-carlo cross-validation generator
scores = []
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

#Assemble classifiers
lda = LinearDiscriminantAnalysis()
svm = svm.SVC(kernel='linear', C=1, random_state=42)
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

#Use scikit-learn pipeline with cross_val_score function
clf = Pipeline([('CSP', csp), ('LDA', lda)])
clf2 = Pipeline([('CSP', csp), ('SVM', svm)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
scores2 = cross_val_score(clf2, epochs_data_train, labels, cv=cv, n_jobs=1)

#And print them
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("LDA Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

#And print them
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("SVM Classification accuracy: %f / Chance level: %f" % (np.mean(scores2),
                                                          class_balance))

# plot some CSP patterns
csp.fit_transform(epochs_data, labels)
csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
plt.show(block=True)

# print(epochs.ch_names)
# epochs.plot_image(picks=['FC5', 'FC3', 'FC1', 'FCz', 'FC2'])
# plt.show(block=True)

#Classification over time.
sfreq = raw.info['sfreq']
w_length = int(sfreq * 0.5)   # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

scores_windows = []

for train_idx, test_idx in cv_split:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    X_test = csp.transform(epochs_data_train[test_idx])

    # fit classifier
    lda.fit(X_train, y_train)

    # running classifier: test classifier on sliding window
    score_this_window = []
    for n in w_start:
        X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
        score_this_window.append(lda.score(X_test, y_test))
    scores_windows.append(score_this_window)

# Plot scores over time
w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
plt.axvline(0, linestyle='--', color='k', label='Onset')
plt.axhline(0.5, linestyle='-', color='k', label='Chance')
plt.xlabel('time (s)')
plt.ylabel('classification accuracy')
plt.title('Classification score over time')
plt.legend(loc='lower right')
plt.show(block=True)