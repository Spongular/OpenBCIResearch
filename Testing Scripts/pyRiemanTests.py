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

# local script imports
import gen_tools
import data_loading
import keras_classifiers

# generic import
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# mne import
import mne
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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# keras imports
from keras import backend
from keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint

#Clear the last session just in case.
backend.clear_session()

#Set the log level to warning to avoid flooding the output window.
mne.set_log_level('WARNING')

#-----------------------------------------------------------------------------------------------------------------------
tmin, tmax = 1., 2
#raw = data_loading.get_single_mi('080', 3)
raw = data_loading.get_all_mi_between(1, 110, 4, ["088", "092", "100"])


event_id = dict(T1=2, T2=3)
# subsample elecs
#picks = picks[::2]

# Apply band-pass filter
raw = gen_tools.preprocess_bandpass(raw, 7., 35.)

events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
picks = pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads', selection=["C3", "Cz", "C4", 'F1'])
#picks = pick_types(
#    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
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
print(cov_data_train.shape)

# #-----------------------------------------------------------------------------------------------------------------------
# #Minimum Distance To Mean Classifier.
# mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
#
# # Use scikit-learn Pipeline with cross_val_score function
# scores = cross_val_score(mdm, cov_data_train, labels, cv=cv, n_jobs=1)
#
# # Printing the results
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# print("MDM Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
#                                                               class_balance))
#
# #-----------------------------------------------------------------------------------------------------------------------
# #Tangent Space Logistic Regression Classifier
# clf = TSclassifier()
# # Use scikit-learn Pipeline with cross_val_score function
# scores = cross_val_score(clf, cov_data_train, labels, cv=cv, n_jobs=1)
#
# # Printing the results
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# print("Tangent space Classification accuracy: %f / Chance level: %f" %
#       (np.mean(scores), class_balance))
#
# #-----------------------------------------------------------------------------------------------------------------------
# # CSP and Logistic Regression Classifier
# lr = LogisticRegression(max_iter=1000)
# csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
#
# clf = Pipeline([('CSP', csp), ('LogisticRegression', lr)])
# scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
#
# # Printing the results
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# print("CSP + LDA Classification accuracy: %f / Chance level: %f" %
#       (np.mean(scores), class_balance))

#-----------------------------------------------------------------------------------------------------------------------
# Feed Forward Neural Network
#Generate model, optimiser and checkpointer
#Tested Models:
#batch_size = 50, model_shape=[128, 256, 512], d_rate=0.2, b_norm=True gave loss: 0.5984 - accuracy: 0.6569 - val_loss: 0.7307 - val_accuracy: 0.5565
#batch_size = 100, model_shape=[1000, 500, 250], d_rate=0.6, b_norm=True gave loss: 0.6724 - accuracy: 0.5744 - val_loss: 0.6837 - val_accuracy: 0.5653
model, opt = keras_classifiers.feedforward_nn(cov_data_train.shape, model_shape=[1000, 500, 250], d_rate=0.6, b_norm=True)
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                               save_best_only=True)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

#Separate the data into train and test sets.
t_data, t_labels = shuffle(cov_data_train, labels, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(t_data, t_labels, test_size=0.6, random_state=1)
y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)

class_weights = {0:1, 1:1}

fittedModel = model.fit(X_train, y_train, batch_size = 100, epochs = 600,
                        verbose = 2, validation_data=(X_val, y_val),
                        class_weight=class_weights, callbacks=[checkpointer])

probs       = model.predict(X_val)
preds       = probs.argmax(axis = -1)
acc         = np.mean(preds == y_val.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

#-----------------------------------------------------------------------------------------------------------------------
# LSTM Neural Network


#-----------------------------------------------------------------------------------------------------------------------
# Convolutional Neural Network


#-----------------------------------------------------------------------------------------------------------------------
#Display the MDM Centroid
mdm = MDM()
mdm.fit(cov_data_train, labels)

fig, axes = plt.subplots(1, 2, figsize=[8, 4])
ch_names = [ch.replace('.', '') for ch in epochs.ch_names]

df = pd.DataFrame(data=mdm.covmeans_[0], index=ch_names, columns=ch_names)
g = sns.heatmap(
    df, ax=axes[0], square=True, cbar=False, xticklabels=2, yticklabels=2)
g.set_title('Mean covariance - T1')

df = pd.DataFrame(data=mdm.covmeans_[1], index=ch_names, columns=ch_names)
g = sns.heatmap(
    df, ax=axes[1], square=True, cbar=False, xticklabels=2, yticklabels=2)
plt.xticks(rotation='vertical')
plt.yticks(rotation='horizontal')
g.set_title('Mean covariance - T2')

# dirty fix
plt.sca(axes[0])
plt.xticks(rotation='vertical')
plt.yticks(rotation='horizontal')
plt.show()

