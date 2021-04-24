
import mne
from mne.io import read_raw_edf

import standard_classifiers
import data_loading
from mne.datasets.eegbci import eegbci

mne.set_log_level('WARNING')

#load our data.
raw = data_loading.get_all_mi_between(2, 3, 4, ["088", "092", "100"])
eegbci.standardize(raw)
raw.set_montage("standard_1020", match_case=False)
raw.rename_channels(lambda s: s.strip("."))

#Filter it.
raw.filter(7., 13., fir_design='firwin', skip_by_annotation='edge') #Bandpass

#classify it.
#standard_classifiers.csp_lda(raw, -1., 4.)
#standard_classifiers.csp_svm(raw, -1., 4.)
#standard_classifiers.csp_lda(raw, -1., 4., ["C3", "Cz", "C4"])

standard_classifiers.pca_lda(raw, -1., 4., ["C3", "Cz", "C4"], n_jobs=2, pca_n_components=32)
standard_classifiers.pca_svm(raw, -1., 4., ["C3", "Cz", "C4"], n_logspace=2, n_jobs=2, pca_n_components=32)
standard_classifiers.pca_knn(raw, -1., 4., ["C3", "Cz", "C4"], max_n_neighbors=10, n_jobs=2, pca_n_components=32)

#standard_classifiers.csp_knn(raw, -1., 4., max_n_neighbors=8, n_jobs=-1, max_csp_components=10)
#For Subjects 1-10 Combined.
#CVGridSearch completed in 41021.308s
# Displaying Results...
# Best score: 0.542
# Best parameters set:
# 	CSP__cov_est: 'concat'
# 	CSP__n_components: 9
# 	CSP__norm_trace: True
# 	KNN__algorithm: 'ball_tree'
# 	KNN__n_neighbors: 7
# 	KNN__weights: 'uniform'
#
# Process finished with exit code 0
