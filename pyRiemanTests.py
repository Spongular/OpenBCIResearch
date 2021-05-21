#https://pyriemann.readthedocs.io/en/latest/auto_examples/motor-imagery/plot_single.html#sphx-glr-auto-examples-motor-imagery-plot-single-py
#NOTE: pyRiemann has issues with Scikit-learn v.24, so in pyRiemann.clustering add...
# try:
#     from sklearn.cluster._kmeans import _init_centroids
# except ImportError:
#     # Workaround for scikit-learn v0.24.0rc1+
#     # See issue: https://github.com/alexandrebarachant/pyRiemann/issues/92
#     from sklearn.cluster import KMeans
#
#     def _init_centroids(X, n_clusters, init, random_state, x_squared_norms):
#         if random_state is not None:
#             random_state = numpy.random.RandomState(random_state)
#         return KMeans(n_clusters=n_clusters)._init_centroids(
#             X,
#             x_squared_norms,
#             init,
#             random_state,
#         )


import gen_tools
import data_loading
import covariance_operations

# generic import
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# mne import
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

# pyriemann import
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances

# sklearn imports
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

#-----------------------------------------------------------------------------------------------------------------------
tmin, tmax = 1., 2
raw = data_loading.get_single_mi('080', 4)
#raw = data_loading.get_all_mi_between(2, 3, 2, ["088", "092", "100"])


event_id = dict(left=2, right=3)
picks = pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
# subsample elecs
picks = picks[::2]

# Apply band-pass filter
raw.filter(7., 35., method='iir', picks=picks)

events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
    verbose=False)
labels = epochs.events[:, -1] - 2

# cross validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)
# get epochs
epochs_data_train = 1e6 * epochs.get_data()

# compute covariance matrices
cov_data_train = Covariances().transform(epochs_data_train)

#-----------------------------------------------------------------------------------------------------------------------
#Minimum Distance To Mean Classifier.
mdm = MDM(metric=dict(mean='riemann', distance='riemann'))

# Use scikit-learn Pipeline with cross_val_score function
scores = cross_val_score(mdm, cov_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("MDM Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                              class_balance))

#-----------------------------------------------------------------------------------------------------------------------
#Tangent Space Logistic Regression Classifier
clf = TSclassifier()
# Use scikit-learn Pipeline with cross_val_score function
scores = cross_val_score(clf, cov_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Tangent space Classification accuracy: %f / Chance level: %f" %
      (np.mean(scores), class_balance))

#-----------------------------------------------------------------------------------------------------------------------
# CSP and Logistic Regression Classifier
lr = LogisticRegression()
csp = CSP(n_components=4, reg='ledoit_wolf', log=True)

clf = Pipeline([('CSP', csp), ('LogisticRegression', lr)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("CSP + LDA Classification accuracy: %f / Chance level: %f" %
      (np.mean(scores), class_balance))

#-----------------------------------------------------------------------------------------------------------------------
#Display the MDM Centroid
mdm = MDM()
mdm.fit(cov_data_train, labels)

fig, axes = plt.subplots(1, 2, figsize=[8, 4])
ch_names = [ch.replace('.', '') for ch in epochs.ch_names]

df = pd.DataFrame(data=mdm.covmeans_[0], index=ch_names, columns=ch_names)
g = sns.heatmap(
    df, ax=axes[0], square=True, cbar=False, xticklabels=2, yticklabels=2)
g.set_title('Mean covariance - hands')

df = pd.DataFrame(data=mdm.covmeans_[1], index=ch_names, columns=ch_names)
g = sns.heatmap(
    df, ax=axes[1], square=True, cbar=False, xticklabels=2, yticklabels=2)
plt.xticks(rotation='vertical')
plt.yticks(rotation='horizontal')
g.set_title('Mean covariance - feets')

# dirty fix
plt.sca(axes[0])
plt.xticks(rotation='vertical')
plt.yticks(rotation='horizontal')
plt.show()