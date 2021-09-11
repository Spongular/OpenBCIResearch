#Source: https://mne.tools/dev/auto_examples/decoding/plot_decoding_csp_eeg.html

#Our imports. I'm being lazy and just importing the whole lot.

#Here, we can set our details for the epoch gathering.
import mne
import numpy as np
import random
import matplotlib.pyplot as plt
from mne import concatenate_raws, events_from_annotations, pick_types, Epochs
from mne.channels import make_standard_montage
from mne.datasets.eegbci import eegbci
from mne.decoding import CSP
from mne.io import read_raw_edf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.model_selection import ShuffleSplit, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
from time import time

mne.set_log_level('WARNING')
tmin, tmax = 0., 4. #This determines the boundaries for the epochs.
#event_dict = dict(hands=2, feet=3) #Define our events.

def csp_lda(max_csp_components=10):
    print("Generating PCA-LDA classifier pipeline...")
    # Assemble classifiers
    lda = LinearDiscriminantAnalysis()
    var = VarianceThreshold(threshold=(.9 * (1 - .9)))
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    # create the pipeline
    clf = Pipeline([('CSP', csp), ('VAR', var), ('LDA', lda)])
    # Form the parameter dictionary
    parameters = {
        'LDA__solver': ('svd', 'lsqr', 'eigen'),
        'CSP__n_components': range(2, max_csp_components + 1),
        'CSP__cov_est': ('concat', 'epoch'),
        'CSP__norm_trace': (True, False),
        'VAR__threshold': ((.9 * (1 - .9)), (.85 * (1 - .85)), (.8 * (1 - .8)),
                           (.75 * (1 - .75)))
    }
    return "CSP-LDA", clf, parameters

#Select the hands & feet tests.
files = []
for a in range(1, 2):
    num = "{0:03}".format(a)
    files += ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R03.edf",
              "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R07.edf",
              "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R11.edf"]

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
epochs_train = epochs.copy()#.crop(tmin = 1., tmax = 2.)
labels = epochs.events[:,-1] -2

print(epochs.get_data().shape)
print(labels.shape)

# Now for a monte-carlo cross-validation generator
scores = []
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()
print(epochs_data.shape)
rand = random.randint(1, 99999)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand)
#cv = ShuffleSplit(10, test_size=0.2, random_state=rand)
#cv_split = cv.split(epochs_data_train)

#Assemble classifiers
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
var = VarianceThreshold(threshold=(.9 * (1 - .9)))
lda = LinearDiscriminantAnalysis()

#Use scikit-learn pipeline with cross_val_score function
#clf = Pipeline([('CSP', csp), ('VAR', var), ('LDA', lda)])
#clf = Pipeline([('CSP', csp), ('LDA', lda)])
name, clf, parameters = csp_lda()
scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("LDA Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                           class_balance))

#This is necessary for multithreading in windows.
bools = list([True, False])

name, clf, parameters = csp_lda()
print("Performing GASearchCV to find optimal parameter set...")
t0 = time()
ga_parameters = {
    'LDA__solver': Categorical(['svd', 'lsqr', 'eigen']),
    'CSP__n_components': Integer(2, 11),
    'CSP__cov_est': Categorical(['concat', 'epoch']),
    #'CSP__norm_trace': Categorical([True, False]),
    'VAR__threshold': Continuous(0, 0.1, distribution='uniform')
}
grid_search = GASearchCV(estimator=clf,
                          cv=cv,
                          scoring='accuracy',
                          param_grid=ga_parameters,
                          n_jobs=2,
                          verbose=True)
grid_search.fit(epochs_data, labels)
print("GASearchCV completed in %0.3fs" % (time() - t0))

# And print out our results.
print("Displaying Results...")
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

print("\n\n")

name, clf, parameters = csp_lda()
print("Performing GridSearchCV to find optimal parameter set...")
t0 = time()
grid_search = GridSearchCV(clf, parameters, n_jobs=2, verbose=0, cv=5)
grid_search.fit(epochs_data, labels)
print("GridSearchCV completed in %0.3fs" % (time() - t0))

# And print out our results.
print("Displaying Results...")
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# #And print them
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# print("LDA Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
#                                                           class_balance))
#
# scores = cross_val_score(clf, epochs_data, labels, cv=cv2, n_jobs=1)
# #And print them
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# print("LDA Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
#                                                           class_balance))

# # plot some CSP patterns
# csp.fit_transform(epochs_data, labels)
# csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
# plt.show(block=True)
#
# # print(epochs.ch_names)
# # epochs.plot_image(picks=['FC5', 'FC3', 'FC1', 'FCz', 'FC2'])
# # plt.show(block=True)
#
# #Classification over time.
# sfreq = raw.info['sfreq']
# w_length = int(sfreq * 0.5)   # running classifier: window length
# w_step = int(sfreq * 0.1)  # running classifier: window step size
# w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)
#
# scores_windows = []
#
# for train_idx, test_idx in cv_split:
#     y_train, y_test = labels[train_idx], labels[test_idx]
#
#     X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
#     X_test = csp.transform(epochs_data_train[test_idx])
#
#     # fit classifier
#     lda.fit(X_train, y_train)
#
#     # running classifier: test classifier on sliding window
#     score_this_window = []
#     for n in w_start:
#         X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
#         score_this_window.append(lda.score(X_test, y_test))
#     scores_windows.append(score_this_window)
#
# # Plot scores over time
# w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin
#
# plt.figure()
# plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
# plt.axvline(0, linestyle='--', color='k', label='Onset')
# plt.axhline(0.5, linestyle='-', color='k', label='Chance')
# plt.xlabel('time (s)')
# plt.ylabel('classification accuracy')
# plt.title('Classification score over time')
# plt.legend(loc='lower right')
# plt.show(block=True)