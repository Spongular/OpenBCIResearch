#Source: https://mne.tools/dev/auto_examples/decoding/plot_decoding_csp_eeg.html

#Our imports. I'm being lazy and just importing the whole lot.

#Here, we can set our details for the epoch gathering.
import mne
import random
from mne import concatenate_raws, events_from_annotations, pick_types, Epochs
from mne.channels import make_standard_montage
from mne.datasets.eegbci import eegbci
from mne.io import read_raw_edf
from sklearn.model_selection import ShuffleSplit, cross_val_score, StratifiedKFold, GridSearchCV
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from moabb.pipelines.utils import FilterBank

mne.set_log_level('WARNING')
tmin, tmax = 0., 4. #This determines the boundaries for the epochs.
#event_dict = dict(hands=2, feet=3) #Define our events.

#Select the hands & feet tests.
files = []
for a in range(1, 2):
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

#The below is from: https://github.com/NeuroTechX/moabb/blob/be1f81220869158ef37e1ab91b0279fe60aeed5b/pipelines/FBCSP.py
# import numpy as np
# from pyriemann.estimation import Covariances
# from pyriemann.spatialfilters import CSP
# from sklearn.feature_selection import SelectKBest, mutual_info_classif
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import make_pipeline
# from sklearn.svm import SVC
#
# from moabb.pipelines.utils import FilterBank
#
#
# parameters = {"C": np.logspace(-2, 2, 10)}
# clf = GridSearchCV(SVC(kernel="linear"), parameters)
# fb = FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4)))
# pipe = make_pipeline(fb, SelectKBest(score_func=mutual_info_classif, k=10), clf)
#
# # this is what will be loaded
# PIPELINE = {
#     "name": "FBCSP + optSVM",
#     "paradigms": ["FilterBankMotorImagery"],
#     "pipeline": pipe,
# }
events, event_id = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

#For each filter range, we filter and append the epoch data onto 'data'
filters = [[8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 35]]
data = []
for filter in filters:
    fmin, fmax = filter
    raw_f = raw.copy().filter(fmin, fmax, method="iir", picks=picks, verbose=False)
    epochs = Epochs(raw_f, events, event_id, tmin, tmax, picks=picks, baseline=None, preload=True, verbose=False)
    data.append(epochs.get_data())
#Now, we rearrange it from (filter, subject, epoch, filter) to (subject, epoch, time, filter)
data = np.array(data).transpose((1, 2, 3, 0))
labels = epochs.events[:,-1] -2
print(data.shape)
print(labels.shape)

rand = random.randint(1, 99999)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand)

parameters = {"C": np.logspace(-2, 2, 10)}
clf = GridSearchCV(SVC(kernel="linear"), parameters)
fb = FilterBank(make_pipeline(Covariances(estimator="oas"), CSP(nfilter=4)))
pipe = make_pipeline(fb, SelectKBest(score_func=mutual_info_classif, k=10), clf)

scores = cross_val_score(pipe, data, labels, cv=cv, n_jobs=1)
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("FBCSP-SVC Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                           class_balance))

