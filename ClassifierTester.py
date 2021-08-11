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
from datetime import datetime
from time import time
import random

# mne import
import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP, Scaler, UnsupervisedSpatialFilter, Vectorizer

# pyriemann import
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances

# sklearn imports
from sklearn import svm
from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression, mutual_info_regression, SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class ClassifierTester:
    """A Class designed to test EEG data against a variety of classifiers.
    This class trains and tests a variety of classifiers on sets of EEG data that vary in size and subject composition.
    The 'Test' function first begins by training/classifying each individual subject, and then a chosen number of
    collections of randomly selected subjects of size equal to 'sub_group_incr', then of size equal to the previous plus
    the increment again, and so on and so forth until the entire group of subjects have been tested at once.
    The intention behind this is to gain a clear perspective as to how the efficacy of classification methods change
    as the breadth of data and variation in subjects increases, so as to determine which methods lend themselves best
    to classifying new or unseen subjects.

        data_source     :   The location/type of data to select. Options are 'physionet' - The PhysioNet Motor
                            movement and imagery database, 'mamem-ssvep' - A public SSVEP dataset, and the two sets
                            of OpenBCI recorded data, 'live-movement' and 'live-imagined'.

        type            :   Indicates the type of event being classifier, in this case either 'movement' or 'imaginary'.
                            Does not apply to SSVEP or the OpenBCI data, only physionet.

        stim_select     :   Refers to the type of movement, real or imaginary, or contains a list of the frequencies
                            in the case of ssvep. e.g. 'lr' and 'hf' are left/right hands and hands/feet respectively.
                            for ssvep, use a list in the form ['6.66', '7.67', ...]

        subj_range      :   A list of two numbers specifying the range of subjects to select from. i.e. [5, 64]
                            will select subjects five through sixty-four inclusive. If None, the entire set of subjects
                            will be used.

        sub_group_incr  :   Indicates the initial size, and incremental increase in size, for the randomised test groups.
                            i.e. if 'sub_group_incr' = 5, the group sizes will be 1, 5, 15, 20 etc. up to the maximum
                            number of subjects. If the maximum number is not a multiple of the increment, it will stop
                            exactly at the max, i.e. if the set of subjects is 1-9, the groups will be 1, 5, 9.

        result_format   :   The format in which results are recorded for the testing. Options include 'acc' for accuracy,
                            'f' for fmeasure.

        gridsearch      :   Indicates whether a gridsearch will be performed on a subset of the chosen subject pool to
                            refine parameters. If 'gridsearch' = None, so search occurs, otherwise gridsearch should be
                            a decimal between 0 and 1 inclusive to indicate the size of the subset to perform the search
                            on. It is recommended to use a small subset so as to save time, as gridsearch is time-intensive.

        file            :   Indicates the filepath to write results to. If 'file' = None, a new file will be generated
                            in the folder 'ClassifierTesterResults' with a name indicating the data tested and the
                            date and time of testing.

        filter_bounds   :   A one or two element list that contains the bounds for the filter. If one element is specified,
                            a highpass filter is performed on the data with the given lower bound. If two elements, then
                            a bandpass filter is used with the bounds specified. Default of [6, 30] is the standard for
                            motor imagery tasks.

        tmin, tmax      :   These indicate the timing bounds for epoching the data, with tmin being the beginning of the
                            epoch, and tmax being the end, in relation to each event marker.

        ch_list         :   A list of channels to be included when epoching data. By default, and empty list will result
                            in the inclusion of all EEG channels. Uses standard 10-20 channel notations, i.e. ['Fp1', 'Cz']
    """
    def __init__(self, data_source='physionet', type='movement', stim_select='lr', subj_range=None,
                 sub_group_incr=5, result_format='acc', gridsearch=None, file=None, filter_bounds=[6., 30.],
                 tmin=0., tmax=4., ch_list=[]):

        type_dict = {('movement', 'lr') : 1,
                     ('imaginary', 'lr') : 2,
                     ('movement', 'hf') : 3,
                     ('imaginary', 'hf') : 4}
        self.sub_data_list = []

        #First, determine where we get our data.
        if data_source == 'physionet':
            #Define our subject range for iteration
            if subj_range is None:
                r1 = 1
                r2 = 110
            elif subj_range[0] > 0 and subj_range[1] < 110 and subj_range[0] < subj_range[1]:
                r1 = subj_range[0]
                r2 = subj_range[1]
            else:
                raise Exception("Error, subj_range is invalid. Ensure it is two values between 1 and 109, with the first being larger")
            print("Selecting {src} data for subjects {start} to {stop}...".format(src=data_source,
                                                                                  start=subj_range[0], stop=subj_range[1]))
            #Now, iterate through the subjects, filter and epoch each, and pair the data and labels in an ordered list.
            for sub in range(r1, r2):
                raw = data_loading.get_single_mi(sub, type_dict[(type, stim_select)])
                if filter_bounds[1] is None:
                    raw = gen_tools.preprocess_highpass(raw, min=filter_bounds[0])
                else:
                    raw = gen_tools.preprocess_bandpass(raw, min=filter_bounds[0], max=filter_bounds[1])
                data, labels = gen_tools.epoch_data(raw, tmin=tmin, tmax=tmax, pick_list=ch_list)
                self.sub_data_list.append([data, labels])
        elif data_source == 'live-movement':
            raise Exception("'live-movement' Not Yet Implemented")
        elif data_source == 'live-imagined':
            raise Exception("'live-imagined' Not Yet Implemented")
        elif data_source == 'mamem-ssvep':
            raise Exception("'mamem-ssvep' Not Yet Implemented")
        else:
            raise Exception("Error: 'data_source' must be one of 'physionet', 'mamem-ssvep', 'live-imagined' or 'live-movement'")

        #Set misc variables
        self.group_incr = sub_group_incr
        self.result_format=result_format
        if file is None:
            #Make our file name.
            self.datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            if data_source == 'physionet':
                filename = "test-results_{src}_{stim}-{type}_{datetime}".format(src=data_source, stim=stim_select,
                                                                                type=type, datetime=self.datetime)
            elif data_source == 'live-imagined':
                filename = "test-results_{src}_{stim}-imaginary_{datetime}".format(src=data_source, stim=stim_select,
                                                                                   datetime=self.datetime)
            elif data_source == 'live-movement':
                filename = "test-results_{src}_{stim}-movement_{datetime}".format(src=data_source, stim=stim_select,
                                                                                   datetime=self.datetime)
            elif data_source == 'mamem-ssvep':
                filename = "test-results_{src}_{stim}_{datetime}".format(src=data_source, stim=stim_select,
                                                                                  datetime=self.datetime)

            #Create and open a new file to record the results of the testing.
            path = "ClassifierTesterResults/{filename}.txt".format(filename=filename)
            self.result_file = open(path, "w+")
            print("Generated new file on path '{path}'".format(path=path))
        else:
            self.result_file = open(file, "w")
            print("File in path '{path}' opened.".format(path=file))

        #Finally, perform the gridsearch method to optimise parameters for the classifiers.
        #For this, we use 20% of the subject pool selected randomly.
        if gridsearch is not None:
            print("Performing Gridsearch on compatible pipelines to find optimal parameters...")
            if gridsearch >= 1 or gridsearch < 0:
                raise Exception("Error: 'gridsearch' must be a value between 0 and 1, or None")
            pool_size = round(len(self.sub_data_list) * gridsearch)
            self.__gridsearch_params(pool_size=pool_size)

        #Summarise the settings here into the results file.
        self.result_file.write("Results for ClassifierTester Class on dataset '{data}'\n".format(data=data_source))
        self.result_file.write("Date/Time: {datetime}\n".format(datetime=self.datetime))
        self.result_file.write("Settings:\n")
        self.result_file.write("    Type = {type1} - {type2}\n".format(type1=type, type2=stim_select))
        if subj_range is None:
            self.result_file.write("    Subject Range = All, Subject Increment = {incr}\n".format(incr=sub_group_incr))
        else:
            self.result_file.write("    Subject Range = {sub_r}, Subject Increment = {incr}\n".format(sub_r=subj_range,
                                                                                                      incr=sub_group_incr))
        self.result_file.write("    Result Format = {format}, Gridsearch = {grd}\n".format(format=result_format,
                                                                                           grd=gridsearch))
        self.result_file.write("    Filter Bounds = {flt}\n".format(flt=filter_bounds))
        self.result_file.write("    tmin = {min}, tmax = {max}\n".format(min=tmin, max=tmax))
        if ch_list == []:
            self.result_file.write("    Channels = All\n")
        else:
            self.result_file.write("    Channels = {chs}\n".format(chs=ch_list))
        return

    ##------------------------------------------------------------------------------------------------------------------
    ##Gridsearch Tools
    ##------------------------------------------------------------------------------------------------------------------

    def __gridsearch_params(self, pool_size):
        #Generate the dictionary.
        self.grid_dict = {}

        #Form the data randomly.
        sub_pool = list(range(0, len(self.sub_data_list)))
        random.shuffle(sub_pool)
        val = sub_pool.pop()
        data = self.sub_data_list[val][0]
        labels = self.sub_data_list[val][1]
        for val in range(0, pool_size - 1):
            val = sub_pool.pop()
            data = np.concatenate((data, self.sub_data_list[val][0]),
                                  axis=0)
            labels = np.concatenate((labels, self.sub_data_list[val][1]),
                                  axis=0)

        #Generate the classifiers to test.
        pipelines = self.__generate_pipelines()

        #Perform a gridsearch for each.
        for pipe in pipelines:
            params = self.__perform_gridsearch(pipe[1], pipe[2], data, labels, n_jobs=2, cross_val=5)

            #Add the parameters to the dictionary using the name as a key.
            self.grid_dict[pipe[0]] = params
        return

    def __generate_pipelines(self):
        #The method will return a tuple of a name, pipeline and the gridsearch parameters.
        #i.e. format is ("name", pipeline, paramers_dict)
        pipelines = []
        return pipelines

    def __perform_gridsearch(self, classifier, parameters, data, labels, n_jobs, verbose=0, cross_val=3):
        # Here, we make use of the CVGridsearch method to check the
        # various combinations of parameters for the best result.
        print("Performing GridSearchCV to find optimal parameter set...")
        t0 = time()
        grid_search = GridSearchCV(classifier, parameters, n_jobs=n_jobs, verbose=verbose, cv=cross_val)
        grid_search.fit(data, labels)
        print("GridSearchCV completed in %0.3fs" % (time() - t0))

        # And print out our results.
        print("Displaying Results...")
        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        return best_parameters

    ##------------------------------------------------------------------------------------------------------------------
    ##Sk-learn Classifiers
    ##------------------------------------------------------------------------------------------------------------------

    def __csp_lda(self, max_csp_components=20):
        print("Generating PCA-LDA classifier pipeline...")
        # Assemble classifiers
        lda = LinearDiscriminantAnalysis()
        csp = CSP(reg=None, log=True)
        # create the pipeline
        clf = Pipeline([('CSP', csp), ('LDA', lda)])
        # Form the parameter dictionary
        parameters = {
            'LDA__solver': ('svd', 'lsqr', 'eigen'),
            'CSP__n_components': range(2, max_csp_components + 1),
            'CSP__cov_est': ('concat', 'epoch'),
            'CSP__norm_trace': (True, False)
        }
        return ("CSP-LDA", clf, parameters)

    def __csp_svm(self, n_logspace=10, max_csp_components=20):
        print("Generating CSP-SVM classifier pipeline...")
        # If our function parameters are invalid, set them right.
        if max_csp_components < 2:  # Components cannot be less than two.
            print("Parameter 'max_csp_components' is below 2, setting to default.")
            max_csp_components = 20
        if n_logspace < 1:  # n_neighbors needs to be positive.
            print("Parameter 'n_neighbors' is below 2, setting to default.")
            n_logspace = 10
        # Assemble classifiers
        svc = svm.SVC()
        csp = CSP(reg=None, log=True)
        # create the pipeline
        clf = Pipeline([('CSP', csp), ('SVC', svc)])
        # Form the parameter dictionary
        parameters = {
            'SVC__C': np.logspace(start=1, stop=10, num=n_logspace, base=10).tolist(),
            # 'SVC__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
            'CSP__n_components': range(2, max_csp_components + 1),
            'CSP__cov_est': ('concat', 'epoch'),
            'CSP__norm_trace': (True, False)
        }
        return ("CSP-SVM", clf, parameters)

    def __csp_knn(self, max_n_neighbors=4, max_csp_components=20):
        print("Generating CSP-KNN classifier pipeline...")
        # If our function parameters are invalid, set them right.
        if max_csp_components < 2:  # Components cannot be less than two.
            print("Parameter 'max_csp_components' is below 2, setting to default.")
            max_csp_components = 20
        if max_n_neighbors < 2:  # n_neighbors needs to be positive.
            print("Parameter 'n_neighbors' is below 2, setting to default.")
            max_n_neighbors = 4
        # Assemble classifiers
        knn = KNeighborsClassifier()
        csp = CSP(reg=None, log=True)
        # create the pipeline
        clf = Pipeline([('CSP', csp), ('KNN', knn)])
        # Form the parameter dictionary
        parameters = {
            'KNN__n_neighbors': range(2, max_n_neighbors + 1),
            'KNN__weights': ('uniform', 'distance'),
            'KNN__algorithm': ('ball_tree', 'kd_tree', 'brute'),
            'CSP__n_components': range(2, max_csp_components + 1),
            'CSP__cov_est': ('concat', 'epoch'),
            'CSP__norm_trace': (True, False)
        }
        return ("CSP-KNN", clf, parameters)

    def __pca_lda(self, pca_n_components=32):
        print("Generating PCA-LDA classifier pipeline...")
        # If our function parameters are invalid, set them right.
        if pca_n_components < 2:  # -1 means all cores, greater than four may lead to issues.
            print("Parameter 'pca_n_components' is below 2, setting to default.")
            pca_n_components = 32
        # Assemble classifiers
        vect = Vectorizer()  # Sklearn requires 2d arrays, so we vectorise them.
        scl = StandardScaler()  # PCA requires standardised data.
        pca = PCA(n_components=pca_n_components)  # This reduces the dimensionality of the data.
        lda = LinearDiscriminantAnalysis()  # This classifies it all.
        # create the pipeline
        clf = Pipeline([('VECT', vect), ('SCL', scl), ('PCA', pca), ('LDA', lda)])
        # Form the parameter dictionary
        parameters = {
            'LDA__solver': ('svd', 'lsqr', 'eigen'),
            'PCA__whiten': (True, False)
        }
        return ("PCA-LDA", clf, parameters)

    def __pca_svm(self, n_logspace=10, pca_n_components=32):
        print("Generating PCA-SVM classifier pipeline...")
        if n_logspace < 2:  # n_neighbors needs to be positive.
            print("Parameter 'n_logspace' is below 2, setting to default.")
            n_logspace = 10
        if pca_n_components < 2:  # -1 means all cores, greater than four may lead to issues.
            print("Parameter 'pca_n_components' is below 2, setting to default.")
            pca_n_components = 32
        # Assemble classifiers
        vect = Vectorizer()  # Sklearn requires 2d arrays, so we vectorise them.
        scl = StandardScaler()  # PCA requires standardised data.
        pca = PCA(n_components=pca_n_components)  # This reduces the dimensionality of the data.
        svc = svm.SVC()  # This classifies it all.
        # create the pipeline
        clf = Pipeline([('VECT', vect), ('SCL', scl), ('PCA', pca), ('SVC', svc)])
        # Form the parameter dictionary, ('KBEST', kbest)
        parameters = {
            'SVC__C': np.logspace(start=1, stop=10, num=n_logspace, base=10).tolist(),
            # 'SVC__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
            'PCA__whiten': (True, False)
        }

        return ("PCA-SVM", clf, parameters)

    def __pca_knn(self, max_n_neighbors=4, pca_n_components=32):
        print("Generating PCA-KNN classifier pipeline...")
        # If our function parameters are invalid, set them right.
        if max_n_neighbors < 2:  # n_neighbors needs to be positive.
            print("Parameter 'n_neighbors' is below 2, setting to default.")
            max_n_neighbors = 4
        if pca_n_components < 2:  # -1 means all cores, greater than four may lead to issues.
            print("Parameter 'pca_n_components' is below 2, setting to default.")
            pca_n_components = 32
        # Assemble classifiers
        vect = Vectorizer()  # Sklearn requires 2d arrays, so we vectorise them.
        scl = StandardScaler()  # PCA requires standardised data.
        pca = PCA(n_components=pca_n_components)  # This reduces the dimensionality of the data.
        knn = KNeighborsClassifier()  # This classifies it all.
        # create the pipeline
        clf = Pipeline([('VECT', vect), ('SCL', scl), ('PCA', pca), ('KNN', knn)])
        # Form the parameter dictionary, ('KBEST', kbest)
        parameters = {
            'KNN__n_neighbors': range(2, max_n_neighbors + 1),
            'KNN__weights': ('uniform', 'distance'),
            'KNN__algorithm': ('ball_tree', 'kd_tree', 'brute'),
            'PCA__whiten': (True, False)
        }
        return ("PCA-KNN", clf, parameters)

    ##------------------------------------------------------------------------------------------------------------------
    ##PyRiemann Classifiers
    ##------------------------------------------------------------------------------------------------------------------


    ##------------------------------------------------------------------------------------------------------------------
    ##Neural Networks
    ##------------------------------------------------------------------------------------------------------------------


    ##------------------------------------------------------------------------------------------------------------------
    ##Public Methods
    ##------------------------------------------------------------------------------------------------------------------
