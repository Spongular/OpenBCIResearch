# https://pyriemann.readthedocs.io/en/latest/auto_examples/motor-imagery/plot_single.html#sphx-glr-auto-examples-motor-imagery-plot-single-py
# NOTE: pyRiemann has issues with Scikit-learn v.24, so in pyRiemann.clustering add...
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

#Filter Bank methods are from:
#https://github.com/NeuroTechX/moabb/blob/be1f81220869158ef37e1ab91b0279fe60aeed5b/pipelines/FBCSP.py

# local script imports
import gen_tools
import data_loading
import keras_classifiers
import LiveBCI

# generic import
import numpy as np
from datetime import datetime
from time import time
import random
from os import listdir, path, remove
import fnmatch
import re
import gc

# mne import
import mne
from mne.decoding import CSP

# pyriemann import
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP as CovCSP

# sklearn imports
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_validate,\
    StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn_genetic.callbacks import ConsecutiveStopping, DeltaThreshold

# keras imports
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend

#MOABB imports
from moabb.pipelines.utils import FilterBank


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

        result_metrics  :   The the metrics by which the results are evaluated and recorded. This is a list of strings
                            naming the metrics, by default only being 'acc' for accuracy. Other metrics include 'f1'
                            for f1 score, 'rec' for recall, 'prec' for precision and 'roc' for ROC-AUC.

        p_select        :   Indicates whether a search will be performed on a subset of the chosen subject pool to
                            refine parameters. If 'p_select' = None, no parameter selection is done and the default
                            parameters are used. In most cases, these are the sk-learn defaults.
                            If 'p_select' = 'gridsearch', a normal gridsearch will be performed using sk-learn's
                            GridSearchCV method. If 'p_select' = 'genetic', the sklearn-genetic-opt method
                            GASearchCV will be used.
                            Gridsearch is exhaustive, and guaranteed to find the best parameter sets, but takes
                            excessive amounts of time. The genetic algorithm is not exhaustive and makes use of early
                            stopping, making it far quicker but not guaranteed to find the best parameter set.

        p_select_frac   :   Indicates the fraction of subjects to perform the parameter selection process on.
                            'p_select_frac' should be a decimal between 0 and 1 inclusive. It is recommended to use a
                            small subset so as to save time, as parameter selection is time-intensive.

        p_n_jobs        :   Indicates the number of threads used in gridsearch/genetic algorithms for parameter
                            selection. Should be an integer from 1 to the total core/thread limit of the system, or
                            -1 to simply use all available.

        p_skip_mdm      :   A boolean indicating whether a parameter search on the MDM pipeline should be performed or
        (Deprecated)        not. MDM is the most time consuming method to train, and parameter searches on it cause a
                            number of warnings and issues, so it is recommended to manually set the parameters to
                            something reasonable. (No Longer Implemented. Can be uncommented.)

        f_path          :   Indicates the filepath to write results to. If 'file' = None, a new file will be generated
                            in the folder 'ClassifierTesterResults' with a name indicating the data tested and the
                            date and time of testing.
                            Ensure that the path ends without a '/', as one is automatically appended during processing
                            of the file path. i.e. a valid f_path value is 'ClassifierTesterResults' and not
                            'ClassifierTesterResults/'

        f_name          :   Indicates the name of the file to write to. If f_name = None, a file name is generated based
                            on the current dataset, stimulus, time and date.

        filter_bounds   :   A one or two element list that contains the bounds for the filter. If one element is specified,
                            a highpass filter is performed on the data with the given lower bound. If two elements, then
                            a bandpass filter is used with the bounds specified. Default of [6, 30] is the standard for
                            motor imagery tasks.

        notch           :   Determines whether a notch filter is used on the data or not. Performs a notch filter at the
                            specified frequency, otherwise does not filter if notch = None.

        tmin, tmax      :   These indicate the timing bounds for epoching the data, with tmin being the beginning of the
                            epoch, and tmax being the end, in relation to each event marker.

        ch_list         :   A list of channels to be included when epoching data. By default, and empty list will result
                            in the inclusion of all EEG channels. Uses standard 10-20 channel notations, i.e. ['Fp1', 'Cz']

        slice_count     :   A number representing how many fragments that an epoch should be 'sliced' into using
                            non-overlapping sliding windows.

        live_layout     :   A string determining the layout of EEG electrodes for the live recorded OpenBCI data.
                            'headband' refers to the electrode set (F1, F2, O1, O2), and 'm_cortex' refers to the
                            set (C3, Cz, C4).

        avg             :   A boolean determining whether the results are presented as an average or an array of scores.

        filter_bank     :   A boolean determining whether the filter bank preprocessing approach is applied or not.
                            Note that only a single classifier implementation, FBCSP-SVM, works with this method.
                            Other classifiers, barring Keras networks, will be omitted.

        random_state    :   The number for the random state of all random selections/operations made in a run.
                            Setting the random state to the same as a previous run should ensure that these selections
                            and processes run identically to the first, barring other differences in parameters.
    """
    #p_skip_mdm=True
    def __init__(self, data_source='physionet', stim_type='movement', stim_select='lr', subj_range=None,
                 result_metrics=["acc"], p_select=None, p_select_frac=0.1, p_n_jobs=2, f_path=None,
                 f_name=None, filter_bounds=(8., 35.), tmin=0., tmax=4., ch_list=[], slice_count=1,
                 live_layout='headband', avg=True, filter_bank=False, random_state=1, notch=None):

        mne.set_log_level('warning')
        if type(filter_bank) is not bool:
            raise Exception("Error: Parameter 'filter_bank' must be of type 'bool'")

        #This is used in selecting types of data from the PhysioNet Dataset.
        type_dict = {('movement', 'lr'): 1,
                     ('imaginary', 'lr'): 2,
                     ('movement', 'hf'): 3,
                     ('imaginary', 'hf'): 4}

        #Misc attributes
        self.data_source = data_source
        self.stim_type = stim_type
        self.stim_select = stim_select
        self.filter_bounds = filter_bounds
        self.notch = notch
        self.tbounds = (tmin, tmax)
        self.ch_list = ch_list
        self.sub_dict = {}
        self.sk_test = False
        self.nn_test = False
        self.avg = avg
        self.subj_range = subj_range
        self.random_state = random_state
        self.filter_bank = filter_bank

        #If no random state is selected, make one.
        if self.random_state is not None and type(self.random_state) != int:
            self.random_state = random.randint(1, 999999)


        # First, determine where we get our data.
        if data_source == 'physionet':
            # Define our subject range for iteration
            if subj_range is None:
                r1 = 1
                r2 = 110
                print("Selecting {src} data for all subjects".format(src=data_source))
            elif subj_range[0] > 0 and subj_range[1] < 111 and subj_range[0] < subj_range[1]:
                r1 = subj_range[0]
                r2 = subj_range[1]
                print("Selecting {src} data for subjects {start} to {stop}...".format(src=data_source,
                                                                                      start=subj_range[0],
                                                                                      stop=subj_range[1]))
            else:
                raise Exception(
                    "Error, subj_range is invalid. Ensure it is two values between 1 and 109, with the first being smaller")
            #Now, we transform the range to a list and remove the bad subjects.
            sub_key_list = list(range(r1, r2))
            ignore_list = [38, 80, 88, 89, 92, 100, 104]
            sub_key_list = list(set(sub_key_list) - set(ignore_list))
            # Now, iterate through the subjects, filter and epoch each, and pair the data and labels in an ordered list.
            for sub in sub_key_list:
                raw = data_loading.get_single_mi(sub, type_dict[(stim_type, stim_select)])
                data, labels = self.__raw_to_data(raw)
                self.sub_dict[sub] = {'data': data,
                                      'labels': labels}

        elif data_source == 'live-movement':
            # Define our subject range for iteration
            if subj_range is None:
                r1 = 1
                r2 = 2
                print("Selecting {src} data for all subjects".format(src=data_source))
            elif subj_range[0] > 0 and subj_range[0] < subj_range[1]:
                r1 = subj_range[0]
                r2 = subj_range[1]
                print("Selecting {src} data for subjects {start} to {stop}...".format(src=data_source,
                                                                                      start=subj_range[0],
                                                                                      stop=subj_range[1]))
            else:
                raise Exception(
                    "Error, subj_range is invalid. Ensure it is two values greater than zero, with the first being smaller")

            #Grab the file names and filter to match what is needed.
            rootpath = 'E:\\PycharmProjects\\OpenBCIResearch\\DataGathering\\LiveRecordings\\MotorResponses\\Movement\\'
            file_paths = listdir(rootpath)
            filt_files = {}
            if live_layout == 'headband':
                for sub in range(r1, r2):
                    filt_files[sub] = fnmatch.filter(file_paths, 'subject{sub}-??_??_????-mm-{stim}*'.format(sub=sub,
                                                                                                 stim=stim_select))
            elif live_layout == 'm_cortex':
                for sub in range(r1, r2):
                    filt_files[sub] = fnmatch.filter(file_paths, 'subject{sub}-??_??_????-m_cortex_electrode_placement-mm-{stim}*'.format(sub=sub,
                                                                                                 stim=stim_select))
            else:
                raise Exception("Error: 'live_layout' must be either 'headband' or 'm_cortex'")

            #Format the file paths and load them through LiveBCI
            for sub, files in filt_files.items():
                file_paths = []
                for file_name in files:
                    file_paths.append(rootpath + file_name)
                dloader = LiveBCI.MotorImageryStimulator(stim_time=4, wait_time=4, stim_count=5, stim_type='lr',
                                                         board=None)
                dloader.load_multiple_data(files=file_paths)
                data, labels = self.__raw_to_data(dloader)
                self.sub_dict[sub] = {'data': data,
                                      'labels': labels}

        elif data_source == 'live-imagined':
            raise Exception("'live-imagined' Not Yet Implemented")
        elif data_source == 'mamem-ssvep':
            raise Exception("'mamem-ssvep' Not Yet Implemented")
        else:
            raise Exception(
                "Error: 'data_source' must be one of 'physionet', 'mamem-ssvep', 'live-imagined' or 'live-movement'")

        if slice_count > 1:
            for key, sub in self.sub_dict.items():
                #for each subject, we perform the slice on their data.
                self.sub_dict[key] = gen_tools.slice_data(sub['data'], sub['labels'], slice_count)

        # Set the metrics
        self.result_metrics = {}
        if 'acc' in result_metrics:
            self.result_metrics['Accuracy'] = 'accuracy'
        if 'rec' in result_metrics:
            self.result_metrics['Recall'] = 'recall'
        if 'prec' in result_metrics:
            self.result_metrics['Precision'] = 'precision'
        if 'f1' in result_metrics:
            self.result_metrics['F1_Score'] = 'f1'
        if 'roc' in result_metrics:
            self.result_metrics['ROC_AUC'] = 'roc_auc'
        if len(self.result_metrics) < 1:
            raise Exception("Error: No valid metric specified in list 'result_metrics'.")

        self.datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        if f_name is None:
            if self.data_source == 'physionet':
                file_name = "test-results_{src}_{stim}-{type}_{datetime}".format(src=self.data_source,
                                                                                stim=self.stim_select,
                                                                                type=self.stim_type,
                                                                                datetime=self.datetime)
            elif self.data_source == 'live-imagined':
                file_name = "test-results_{src}_{stim}-imaginary_{datetime}".format(src=self.data_source,
                                                                                   stim=self.stim_select,
                                                                                   datetime=self.datetime)
            elif self.data_source == 'live-movement':
                file_name = "test-results_{src}_{stim}-movement_{datetime}".format(src=self.data_source,
                                                                                  stim=self.stim_select,
                                                                                  datetime=self.datetime)
            elif self.data_source == 'mamem-ssvep':
                file_name = "test-results_{src}_{stim}_{datetime}".format(src=self.data_source, stim=self.stim_select,
                                                                         datetime=self.datetime)
            else:
                raise Exception("Error: 'data_source' must be a string matching either 'physionet', 'live-imagined', "
                                "'live-movement' or 'mamem-ssvep'.")
        else:
            file_name = f_name
        if f_path is None or type(f_path) != str:
            # Create and open a new file to record the results of the testing.
            path = "ClassifierTesterResults/{file_name}.txt".format(file_name=file_name)
            self.result_file = open(path, "w+")
            print("Generated new file on path '{path}'".format(path=path))
        else:
            path = "{file_path}/{file_name}.txt".format(file_path=f_path, file_name=file_name)
            self.result_file = open(path, "w")
            print("File in path '{path}' opened.".format(path=path))

        # Finally, perform the gridsearch method to optimise parameters for the classifiers.
        # For this, we use 20% of the subject pool selected randomly.
        self.sk_dict = {}
        self.nn_dict = {}
        self.p_select = p_select
        self.p_n_jobs = p_n_jobs
        #self.p_skip_mdm = p_skip_mdm

        self.p_select_frac = p_select_frac
        if self.p_select is not None:
            if self.p_select == 'gridsearch':
                print("Performing Gridsearch on compatible pipelines to find optimal parameters...")
            elif self.p_select == 'genetic':
                print("Performing Genetic search on compatible pipelines to find optimal parameters...")
            if p_select_frac > 1 or p_select_frac <= 0:
                raise Exception("Error: 'p_select_frac' must be a value between 0 and 1, or None")
            if p_select_frac is None:
                p_select_frac = 1
            self.__gridsearch_params(test_frac=p_select_frac, n_jobs=self.p_n_jobs)
        else:
            #Without a gridsearch, we just fill the class dictionary with the classifiers set to default.
            pipelines = self.__generate_pipelines()
            for pipe in pipelines:
                self.sk_dict[pipe[0]] = pipe[1]

        #This flag indicates that the classifiers have not been loaded.
        #The gridsearch dict is checked for the non-nn classifiers first.
        if len(self.sk_dict) > 0:
            self.sk_class_loaded = True
        else:
            self.sk_class_loaded = False
        self.nn_class_loaded = False


        # Summarise the settings here into the results file.
        self.__print("Results for ClassifierTester Class on dataset '{data}'\n".format(data=data_source))
        self.__print("Date/Time: {datetime}\n".format(datetime=self.datetime))
        self.__print("Settings:\n")
        self.__print("    Type = {type1} - {type2}\n".format(type1=stim_type, type2=stim_select))
        if subj_range is None:
            self.__print("    Subject Range = All\n")
        else:
            self.__print("    Subject Range = {sub_r}\n".format(sub_r=subj_range))
        self.__print("    Result Metrics = {format}\n".format(format=result_metrics))
        self.__print("    Gridsearch = {grd}".format(grd=p_select))
        self.__print("    Filter Bounds = {flt}\n".format(flt=filter_bounds))
        self.__print("    Notch Filter = {flt}\n".format(flt=notch))
        self.__print("    tmin = {min}, tmax = {max}\n".format(min=tmin, max=tmax))
        if ch_list == []:
            self.__print("    Channels = All\n")
        else:
            self.__print("    Channels = {chs}\n".format(chs=ch_list))
        self.__print("    Random State = {rand}\n".format(rand=random_state))
        return

    def __del__(self):
        print("Closing and saving file: {file}".format(file=self.result_file.name))
        self.result_file.close()
        return

    ##------------------------------------------------------------------------------------------------------------------
    ##Gridsearch and Preprocessing Tools
    ##------------------------------------------------------------------------------------------------------------------

    def __raw_to_data(self, raw_source):
        if self.data_source == 'physionet' and isinstance(raw_source, mne.io.edf.edf.RawEDF):
            return self.__process_physionet(raw_source)
        elif (self.data_source == 'live-movement' or self.data_source == 'live-imagined') \
                and isinstance(raw_source, LiveBCI.MotorImageryStimulator):
            return self.__process_live(raw_source)
        else:
            raise Exception("Error: Parameter 'data_source' must be either 'physionet', "
                            "'live-movement' or 'live-imagined'")

    def __process_physionet(self, raw_source):
        if self.filter_bank is False:
            if self.filter_bounds[1] is None:
                raw = gen_tools.preprocess_highpass(raw_source, min=self.filter_bounds[0], notch_val=self.notch)
            else:
                if self.notch is not None:
                    print("Performing Notch filter at {n}hz".format(n=self.notch))
                    raw_source.notch_filter(self.notch, filter_length='auto', phase='zero')
                raw = gen_tools.preprocess_bandpass(raw_source, min=self.filter_bounds[0], max=self.filter_bounds[1])
            data, labels, epochs = gen_tools.epoch_data(raw, tmin=self.tbounds[0], tmax=self.tbounds[1],
                                                        pick_list=self.ch_list)
            del epochs
            return data, labels
        else:
            filters = [[8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 35]]
            data, labels = gen_tools.process_filter_bank(raw_source, filters, tmin=self.tbounds[0], tmax=self.tbounds[1],
                                                         pick_list=self.ch_list)
            return data, labels

    def __process_live(self, raw_source):
        if self.filter_bank is False:
            if self.notch is not None:
                print("Performing Notch filter at {n}hz".format(n=self.notch))
                raw_source.raw.notch_filter(self.notch, filter_length='auto', phase='zero')
            raw_source.raw = gen_tools.preprocess_bandpass(raw_source.raw, min=self.filter_bounds[0],
                                                           max=self.filter_bounds[1])
            raw_source.eeg_to_epochs(tmin=self.tbounds[0], tmax=self.tbounds[1], event_dict=dict(T1=2, T2=3), stim_ch='STI001')
            epochs = raw_source.epochs
            data = epochs.get_data()
            labels = epochs.events[:, -1] - 2
        else:
            raw = raw_source.raw
            events = mne.find_events(raw=raw, stim_channel='STI001')
            event_dict = dict(T1=2, T2=3)
            filters = [[8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 35]]
            data, labels = gen_tools.process_filter_bank(raw, filters, tmin=self.tbounds[0], tmax=self.tbounds[1],
                                                         pick_list=self.ch_list, events=events, event_id=event_dict)
        return data, labels


    def __gridsearch_params(self, test_frac, n_jobs=2):
        # Generate the dictionary.
        self.sk_dict = {}

        # Form the data randomly.
        sub_pool = list(self.sub_dict.keys())
        val = sub_pool.pop()
        data = self.sub_dict[val]['data']
        labels = self.sub_dict[val]['labels']
        for sub in sub_pool:
            data = np.concatenate((data, self.sub_dict[sub]['data']),
                                  axis=0)
            labels = np.concatenate((labels, self.sub_dict[sub]['labels']),
                                    axis=0)

        #To get a representative split of the data, let's use shufflesplit and only take the test split.
        if test_frac >= 1:
            x_test = data
            y_test = labels
        else:
            ss = StratifiedShuffleSplit(n_splits=1, test_size=float(test_frac), random_state=self.random_state)
            x_test, y_test = None, None
            first = True
            for train_index, test_index in ss.split(data, labels):
                if not first: continue
                x_test = data[test_index]
                y_test = labels[test_index]
                first = False
        del data, labels

        # Generate the classifiers to test.
        pipelines = self.__generate_pipelines()

        # Perform a gridsearch for each.
        t0 = time()
        for pipe in pipelines:
            # if pipe[0] == 'MDM' and self.p_skip_mdm: #We don't perform gridsearch on MDM as it is too time consuming.
            #     self.sk_dict[pipe[0]] = pipe[1]
            #     continue
            self.__print("\nPerforming parameter search on pipeline: {pipe}\n".format(pipe=pipe[0]))
            if self.p_select == 'gridsearch':
                grid = self.__perform_gridsearch(pipe[1], pipe[2], x_test, y_test, n_jobs=n_jobs, cross_val=3)
            elif self.p_select == 'genetic':
                grid = self.__perform_genetic(pipe[1], pipe[2], x_test, y_test, n_jobs=n_jobs, cross_val=3)
            else:
                raise Exception("Error: parameter 'p_select' must be either None, 'gridsearch' or 'genetic'")

            # Add the best estimator to the dictionary using the name as a key.
            self.sk_dict[pipe[0]] = grid.best_estimator_
        t1 = time() - t0
        self.__print("\nAll Parameter Searches Completed.\nTime Elapsed: {n}\n\n".format(n=t1))
        return

    def __perform_gridsearch(self, classifier, parameters, data, labels, n_jobs=2, verbose=0, cross_val=5):
        # Here, we make use of the CVGridsearch method to check the
        # various combinations of parameters for the best result.
        self.__print("\nPerforming GridSearchCV to find optimal parameter set...\n")
        t0 = time()
        grid_search = GridSearchCV(classifier, parameters, n_jobs=n_jobs, verbose=verbose, cv=cross_val)
        grid_search.fit(data, labels)
        self.__print("GridSearchCV completed in %0.3fs\n" % (time() - t0))

        # And print out our results.
        self.__print("Displaying Results...\n")
        self.__print("Best score: %0.3f\n" % grid_search.best_score_)
        self.__print("Best parameters set:\n")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            self.__print("\t%s: %r\n" % (param_name, best_parameters[param_name]))
        return grid_search

    def __perform_genetic(self, classifier, parameters, data, labels, n_jobs=2, verbose=False, cross_val=5):
        #First, we make the validators and callbacks
        cv = StratifiedKFold(n_splits=cross_val, shuffle=True, random_state=self.random_state)
        #callback = ConsecutiveStopping(generations=5, metric='fitness')
        callback = DeltaThreshold(threshold=0.001, generations=3, metric='fitness_min')

        #Perform the genetic search
        self.__print("\nPerforming GASearchCV to find optimal parameter set...\n")
        t0 = time()
        genetic_search = GASearchCV(estimator=classifier,
                                 cv=cv,
                                 scoring='accuracy',
                                 param_grid=parameters,
                                 n_jobs=n_jobs,
                                 verbose=verbose)
        genetic_search.fit(data, labels, callbacks=callback)
        self.__print("GASearchCV completed in %0.3fs\n" % (time() - t0))

        #Now, print the results
        self.__print("Displaying Results...\n")
        self.__print("Best score: %0.3f\n" % genetic_search.best_score_)
        self.__print("Best parameters set:\n")
        best_parameters = genetic_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            self.__print("\t%s: %r\n" % (param_name, best_parameters[param_name]))
        return genetic_search

    ##------------------------------------------------------------------------------------------------------------------
    ##General Tools
    ##------------------------------------------------------------------------------------------------------------------

    def __generate_neural_networks(self, data_shape):
        #This method will return a tuple of name, compiled NN model and callbacks
        #i.e. format is ("name", model, fit_details)
        models = [self.__eegnet(data_shape=data_shape),
                  self.__fusion_eegnet(data_shape=data_shape),
                  self.__deep_convnet(data_shape=data_shape),
                  self.__shallow_convnet(data_shape=data_shape)]
        return models

    def __generate_pipelines(self):
        # The method will return a tuple of a name, pipeline and the gridsearch parameters.
        # i.e. format is ("name", pipeline, params_dict)
        if self.filter_bank:
            pipelines = [self.__fbcsp_svm()]
        else:
            pipelines = [self.__csp_knn(),
                         self.__csp_svm(),
                         self.__csp_lda(),
                         self.__mdm(),
                         self.__ts_lr(),
                         self.__covcsp_lda(),
                         self.__covcsp_lr()]
        return pipelines

    def __generate_pipelines_from_dict(self, parameter_dict):
        #The parameter dict should be a nested dictionary, of the form dict['classifier'][parameter].
        #As each classifier has a different generating function, let's just use a list of if statements.
        #Not exactly pretty looking, but it works, and this isn't an intensive operation so efficiency isn't important.
        pipelines = []
        for classifier_name, parameters in parameter_dict.items():
            if self.filter_bank is False:
                if classifier_name == 'CSP-LDA':
                    pipe = self.__csp_lda(params=parameters)
                elif classifier_name == 'CSP-SVM':
                    pipe = self.__csp_svm(params=parameters)
                elif classifier_name == 'CSP-KNN':
                    pipe = self.__csp_knn(params=parameters)
                elif classifier_name == 'MDM':
                    pipe = self.__mdm(params=parameters)
                elif classifier_name == 'TS-LR':
                    pipe = self.__ts_lr(params=parameters)
                elif classifier_name == 'CovCSP-LDA':
                    pipe = self.__covcsp_lda(params=parameters)
                elif classifier_name == 'CovCSP-LR':
                    pipe = self.__covcsp_lr(params=parameters)
                else:
                    raise Exception("Error: Parameter 'parameter_dict' contains an invalid key. Keys can only include"
                                    "'CSP-LDA', 'CSP-SVM', 'CSP-KNN', 'MDM', 'TS-LR', 'CovCSP-LDA' or 'CovCSP-LR'.")
            else:
                if classifier_name == 'FBCSP-SVM':
                    pipe = self.__fbcsp_svm(params=parameters)
                else:
                    raise Exception("Error: Parameter 'parameter_dict' contains an invalid key. Keys can only include"
                                    "'FBCSP-SVM' while class parameter 'filter_bank' is True.")
            pipelines.append(pipe)
        return pipelines

    def initialise_sklearn_classifiers(self, parameter_dict=None):
        #This will create sklearn-based classifiers with default parameters
        #that are determined by examples provided in the documentation for
        #MNE and PyRiemann libraries.
        if len(self.sk_dict.keys()) > 0:
            cont = self.__await_yesno(q_string="Warning: SKLearn classifiers are already initialised, do you wish to "
                                               "overwrite them (y/n)?")
            if cont is False:
                return self.sk_dict

        if parameter_dict is None:
            if self.filter_bank is False:
                pipelines = [self.__csp_knn(),
                             self.__csp_svm(),
                             self.__csp_lda(),
                             self.__mdm(),
                             self.__ts_lr(),
                             self.__covcsp_lda(),
                             self.__covcsp_lr()]
            else:
                pipelines = [self.__fbcsp_svm()]
        else:
            pipelines = self.__generate_sklearn_pipelines_from_dict(parameter_dict)

        for pipe in pipelines:
            self.sk_dict[pipe[0]] = pipe[1]
        return self.sk_dict

    def initialise_neural_networks(self, d_shape=None, d_fourth_axis=1):
        #This will construct compiled models set to defaults as determined
        #by the EEGNet and Fusion EEGNet documentation.
        self.nn_dict = {}

        if d_shape is None:
            data_shape = self.sub_dict[list(self.sub_dict.keys())[0]]['data'].shape + (d_fourth_axis,)
        else:
            data_shape = d_shape
        models = [self.__eegnet(data_shape=data_shape),
                  self.__fusion_eegnet(data_shape=data_shape),
                  self.__deep_convnet(data_shape=data_shape),
                  self.__shallow_convnet(data_shape=data_shape)]
        for model in models:
            self.nn_dict[model[0]] = (model[1], model[2])
        return self.nn_dict

    def __print_average_results(self, results):
        #Here we get the average for each metric and print it.
        for name in results.keys():
            self.__print("Classifier: {n}\n".format(n=name))
            for metric in results[name].keys():
                self.__print("{m} = {r}\n".format(m=metric, r=np.mean(results[name][metric])))
                self.__print("{m}_std = {r}\n".format(m=metric, r=np.std(results[name][metric])))
            self.__print("\n")
        return

    def __print_non_average_results(self, results):
        #Here, we need to print each run of data.
        m = list(results)[0]
        n = list(results[m])[0]
        length = len(results[m][n])

        for x in range(0, length):
            self.__print("Value: {n}\n".format(n=x + 1))
            for name in results.keys():
                self.__print("Classifier: {n}\n".format(n=name))
                for metric in results[name].keys():
                    self.__print("{m} = {r}\n".format(m=metric, r=results[name][metric][x]))
                self.__print("\n")
        return

    def __print_results(self, results):
        #Here we get the average for each metric and print it.
        for name in results.keys():
            self.__print("Classifier: {n}\n".format(n=name))
            for metric in results[name].keys():
                self.__print("{m} = {r}\n".format(m=metric, r=results[name][metric]))
                self.__print("{m}_std = {r}\n".format(m=metric, r=np.std(results[name][metric])))
            self.__print("\n")
        return

    def __print(self, string):
        print(string)
        self.result_file.write(string)
        return

    #Gives a particular question and
    def __await_yesno(self, q_string='(y/n)?'):
        val_input = False
        ret_val = False
        while not val_input:
            inp = input(q_string)
            if re.match(pattern='y|Y', string=inp):
                ret_val = True
                val_input = True
            elif re.match(pattern='n|N', string=inp):
                ret_val = False
                val_input = True
            else:
                print("Error: Invalid response, please respond with either 'y' or 'n' in upper or lower case.")
        return ret_val

    def toggle_callbacks(self):
        if self.callbacks is True:
            self.callbacks = False
        else:
            self.callbacks = True
        return

    ##------------------------------------------------------------------------------------------------------------------
    ##Sk-learn Classifiers
    ##------------------------------------------------------------------------------------------------------------------

    # Constructed from sources:
    #   https://mne.tools/dev/auto_examples/decoding/decoding_csp_eeg.html
    #   https://scikit-learn.org/stable/supervised_learning.html

    def __csp_lda(self, max_csp_components=10, params=None):
        print("Generating PCA-LDA classifier pipeline...")
        # Assemble classifiers
        if params is None:
            lda = LinearDiscriminantAnalysis()
            var = VarianceThreshold(threshold=0)
            csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        elif type(params) is dict:
            lda = LinearDiscriminantAnalysis(solver=params['LDA__solver'])
            var = VarianceThreshold(threshold=params['VAR__threshold'])
            csp = CSP(n_components=params['CSP__n_components'], reg=None, log=True, norm_trace=False,
                      cov_est=params['CSP__cov_est'])
        else:
            raise Exception("Error: Parameter 'params' must be of type 'dict'.")

        # create the pipeline
        clf = Pipeline([('CSP', csp), ('VAR', var), ('LDA', lda)])
        # Form the parameter dictionary
        if self.p_select == 'genetic':
            parameters = {
                'LDA__solver': Categorical(['svd', 'lsqr', 'eigen']),
                'CSP__n_components': Integer(2, (max_csp_components + 1)),
                'CSP__cov_est': Categorical(['concat', 'epoch']),
                'VAR__threshold': Categorical([0, 0.001, 0.0025, 0.005, 0.0075, 0.01])
            }
        else:
            parameters = {
                'LDA__solver': ('svd', 'lsqr', 'eigen'),
                'CSP__n_components': range(2, max_csp_components + 1),
                'CSP__cov_est': ('concat', 'epoch'),
                'VAR__threshold': np.linspace(start=0, stop=0.1, num=15)
            }
        return "CSP-LDA", clf, parameters

    def __csp_svm(self, n_logspace=10, max_csp_components=10, params=None):
        print("Generating CSP-SVM classifier pipeline...")
        # If our function parameters are invalid, set them right.
        if max_csp_components < 2:  # Components cannot be less than two.
            print("Parameter 'max_csp_components' is below 2, setting to default.")
            max_csp_components = 20
        if n_logspace < 1:  # n_neighbors needs to be positive.
            print("Parameter 'n_neighbors' is below 2, setting to default.")
            n_logspace = 10
        # Assemble classifiers
        if params is None:
            svc = svm.SVC()
            var = VarianceThreshold(threshold=0)
            csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        elif type(params) is dict:
            svc = svm.SVC(kernel=params['SVC__kernel'], C=params['SVC__C'])
            var = VarianceThreshold(threshold=params['VAR__threshold'])
            csp = CSP(n_components=params['CSP__n_components'], reg=None, log=True, norm_trace=False,
                      cov_est=params['CSP__cov_est'])
        else:
            raise Exception("Error: Parameter 'params' must be of type 'dict'.")
        # create the pipeline
        clf = Pipeline([('CSP', csp), ('VAR', var), ('SVC', svc)])
        # Form the parameter dictionary
        if self.p_select == 'genetic':
            parameters = {
                'SVC__C': Categorical([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]),
                'SVC__kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
                'CSP__n_components': Integer(2, (max_csp_components + 1)),
                'CSP__cov_est': Categorical(['concat', 'epoch']),
                'VAR__threshold': Categorical([0, 0.001, 0.0025, 0.005, 0.0075, 0.01])
            }
        else:
            parameters = {
                'SVC__C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
                'SVC__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                'CSP__n_components': range(2, max_csp_components + 1),
                'CSP__cov_est': ('concat', 'epoch'),
                'VAR__threshold': np.linspace(start=0, stop=0.1, num=15)
            }
        return "CSP-SVM", clf, parameters

    def __csp_knn(self, max_n_neighbors=4, max_csp_components=10, params=None):
        print("Generating CSP-KNN classifier pipeline...")
        # If our function parameters are invalid, set them right.
        if max_csp_components < 2:  # Components cannot be less than two.
            print("Parameter 'max_csp_components' is below 2, setting to default.")
            max_csp_components = 20
        if max_n_neighbors < 2:  # n_neighbors needs to be positive.
            print("Parameter 'n_neighbors' is below 2, setting to default.")
            max_n_neighbors = 4
        # Assemble classifiers
        if params is None:
            knn = KNeighborsClassifier()
            var = VarianceThreshold(threshold=0)
            csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        elif type(params) is dict:
            knn = KNeighborsClassifier(n_neighbors=params['KNN__n_neighbors'], weights=params['KNN__weights'])
            var = VarianceThreshold(threshold=params['VAR__threshold'])
            csp = CSP(n_components=params['CSP__n_components'], reg=None, log=True, norm_trace=False,
                      cov_est=params['CSP__cov_est'])
        else:
            raise Exception("Error: Parameter 'params' must be of type 'dict'.")
        # create the pipeline
        clf = Pipeline([('CSP', csp), ('VAR', var), ('KNN', knn)])
        # Form the parameter dictionary
        if self.p_select == 'genetic':
            parameters = {
                'KNN__n_neighbors': Integer(2, (max_n_neighbors + 1)),
                'KNN__weights': Categorical(['uniform', 'distance']),
                'KNN__algorithm': Categorical(['ball_tree', 'kd_tree', 'brute']),
                'CSP__n_components': Integer(2, (max_csp_components + 1)),
                'CSP__cov_est': Categorical(['concat', 'epoch']),
                'VAR__threshold': Categorical([0, 0.001, 0.0025, 0.005, 0.0075, 0.01])
            }
        else:
            parameters = {
                'KNN__n_neighbors': range(2, max_n_neighbors + 1),
                'KNN__weights': ('uniform', 'distance'),
                'KNN__algorithm': ('ball_tree', 'kd_tree', 'brute'),
                'CSP__n_components': range(2, max_csp_components + 1),
                'CSP__cov_est': ('concat', 'epoch'),
                'VAR__threshold': np.linspace(start=0, stop=0.1, num=15)
            }
        return "CSP-KNN", clf, parameters

    ##------------------------------------------------------------------------------------------------------------------
    ##PyRiemann Classifiers
    ##------------------------------------------------------------------------------------------------------------------

    # Constructed from sources:
    #   https://pyriemann.readthedocs.io/en/latest/api.html
    #   https://github.com/NeuroTechX/moabb/tree/master/pipelines
    #   https://neurotechx.github.io/eeg-notebooks/auto_examples/visual_ssvep/02r__ssvep_decoding.html

    #'mcd' is not used in the estimator search as it is far too time-intensive.
    def __mdm(self, params=None):
        print("Generating MDM classifier pipeline...")
        # Assemble classifier
        if params is None:
            cov = Covariances(estimator='cov')
            mdm = MDM(metric='riemann')
        elif type(params) is dict:
            cov = Covariances(estimator=params['COV__estimator'])
            mdm = MDM(metric=params['MDM__metric'])
        else:
            raise Exception("Error: Parameter 'params' must be of type 'dict'.")
        # create the pipeline
        clf = Pipeline([('COV', cov), ('MDM', mdm)])
        # Form the parameter dictionary
        if self.p_select == 'genetic':
            parameters = {
                'COV__estimator': Categorical(['cov', 'scm', 'lwf', 'oas', 'corr']),
                'MDM__metric': Categorical(['riemann', 'logeuclid', 'euclid', 'wasserstein'])
            }
        else:
            parameters = {
                'COV__estimator': ('cov', 'scm', 'lwf', 'oas', 'corr'),
                'MDM__metric': ('riemann', 'logeuclid', 'euclid', 'wasserstein')
            }
        return "MDM", clf, parameters

    def __ts_lr(self, params=None):
        print("Generating TS-LR classifier pipeline...")
        if params is None:
            cov = Covariances()
            ts = TangentSpace()
            lr = LogisticRegression(max_iter=100000)
        elif type(params) is dict:
            cov = Covariances(estimator=params['COV__estimator'])
            ts = TangentSpace(metric=params['TS__metric'])
            lr = LogisticRegression(max_iter=100000, C=params['LR__C'])
        else:
            raise Exception("Error: Parameter 'params' must be of type 'dict'.")
        clf = Pipeline([('COV', cov), ('TS', ts), ('LR', lr)])
        # Form the parameter dictionary
        if self.p_select == 'genetic':
            parameters = {
                'COV__estimator': Categorical(['cov', 'scm', 'lwf', 'oas', 'corr']),
                'LR__C': Categorical([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]),
                'TS__metric': Categorical(['riemann', 'logeuclid', 'euclid', 'wasserstein'])
            }
        else:
            parameters = {
                'COV__estimator': ('cov', 'scm', 'lwf', 'oas', 'corr'),
                'LR__C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
                'TS__metric': ('riemann', 'logeuclid', 'euclid', 'wasserstein')
            }
        return "TS-LR", clf, parameters

    def __covcsp_lda(self, max_n_filters=10, params=None):
        print("Generating CovCSP-LDA classifier pipeline...")
        # If our function parameters are invalid, set them right.
        if max_n_filters < 2:  # Components cannot be less than two.
            print("Parameter 'max_n_filters' is below 2, setting to default.")
            max_n_filters = 10
        if params is None:
            cov = Covariances()
            csp = CovCSP(nfilter=4, log=True)
            lda = LinearDiscriminantAnalysis()
        elif type(params) is dict:
            cov = Covariances(estimator=params['COV__estimator'])
            csp = CovCSP(nfilter=params['CSP__nfilter'], log=True, metric=params['CSP__metric'])
            lda = LinearDiscriminantAnalysis(solver=params['LDA__solver'])
        else:
            raise Exception("Error: Parameter 'params' must be of type 'dict'.")
        clf = Pipeline([('COV', cov), ('CSP', csp), ('LDA', lda)])
        # Form the parameter dictionary
        if self.p_select == 'genetic':
            parameters = {
                'COV__estimator': Categorical(['cov', 'scm', 'lwf', 'oas', 'corr']),
                'CSP__nfilter': Integer(2, max_n_filters),
                'CSP__metric': Categorical(['riemann', 'logeuclid', 'euclid', 'wasserstein']),
                'LDA__solver': Categorical(['svd', 'lsqr', 'eigen'])
            }
        else:
            parameters = {
                'COV__estimator': ('cov', 'scm', 'lwf', 'oas', 'corr'),
                'CSP__nfilter': range(2, max_n_filters),
                'CSP__metric': ('riemann', 'logeuclid', 'euclid', 'wasserstein'),
                'LDA__solver': ('svd', 'lsqr', 'eigen')
            }
        return "CovCSP-LDA", clf, parameters

    def __covcsp_lr(self, max_n_filters=10, params=None):
        print("Generating CovCSP-LR classifier pipeline...")
        # If our function parameters are invalid, set them right.
        if max_n_filters < 2:  # Components cannot be less than two.
            print("Parameter 'max_n_filters' is below 2, setting to default.")
            max_n_filters = 10
        if params is None:
            cov = Covariances()
            csp = CovCSP(nfilter=4, log=True)
            lr = LogisticRegression(max_iter=100000)
        elif type(params) is dict:
            cov = Covariances(estimator=params['COV__estimator'])
            csp = CovCSP(nfilter=params['CSP__nfilter'], log=True, metric=params['CSP__metric'])
            lr = LogisticRegression(max_iter=100000, C=params['LR__C'])
        else:
            raise Exception("Error: Parameter 'params' must be of type 'dict'.")
        clf = Pipeline([('COV', cov), ('CSP', csp), ('LR', lr)])
        # Form the parameter dictionary
        if self.p_select == 'genetic':
            parameters = {
                'COV__estimator': Categorical(['cov', 'scm', 'lwf', 'oas', 'corr']),
                'CSP__nfilter': Integer(2, max_n_filters),
                'LR__C': Categorical([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]),
                'CSP__metric': Categorical(['riemann', 'logeuclid', 'euclid', 'logdet', 'wasserstein'])
            }
        else:
            parameters = {
                'COV__estimator': ('cov', 'scm', 'lwf', 'oas', 'corr'),
                'LR__C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
                'CSP__nfilter': range(2, max_n_filters),
                'CSP__metric': ('riemann', 'logeuclid', 'euclid', 'logdet', 'wasserstein')
            }
        return "CovCSP-LR", clf, parameters

    ##------------------------------------------------------------------------------------------------------------------
    ##Filter-Bank and MOABB Classifiers
    ##------------------------------------------------------------------------------------------------------------------

    #For filter bank, we need the data in the form (trial, channel, times, filter).
    def __fbcsp_svm(self, params=None):
        #This one is pretty simple, and we don't change up many parameters.
        #Additionally, the
        if params is None:
            fb = FilterBank(make_pipeline(Covariances(estimator="oas"), CovCSP(nfilter=4)))
            kb = SelectKBest(score_func=mutual_info_classif, k=10)
            svc = svm.SVC(kernel="rbf")
        elif type(params) is dict:
            fb = FilterBank(make_pipeline(Covariances(estimator="oas"), CovCSP(nfilter=4)))
            kb = SelectKBest(score_func=mutual_info_classif, k=10)
            svc = svm.SVC(kernel=params['SVC__kernel'], C=params['SVC__C'])
        else:
            raise Exception("Error: Parameter 'params' must be of type 'dict'.")
        clf = Pipeline([('FB', fb), ('KB', kb), ('SVC', svc)])
        if self.p_select == 'genetic':
            parameters = {
                'SVC__C': Categorical([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]),
                'SVC__kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid'])
            }
        else:
            parameters = {
                'SVC__C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
                'SVC__kernel': ('linear', 'poly', 'rbf', 'sigmoid')
            }
        return "FBCSP-SVM", clf, parameters

    ##------------------------------------------------------------------------------------------------------------------
    ##Neural Networks
    ##------------------------------------------------------------------------------------------------------------------

    #An Accurate EEGNet-based Motor-Imagery Brain–Computer Interface for Low-Power Edge Computing
    #Available at: https://arxiv.org/abs/2004.00077
    def __eegnet(self, data_shape, dropout=0.5, n_classes=2, lr_scheduler=True, check_p=True, e_stop=False,
                 l_rate=0.01):

        #First, construct the model and compile it.
        model, opt = keras_classifiers.convEEGNet(input_shape=data_shape, chan=data_shape[1], n_classes=n_classes,
                                                  d_rate=dropout, first_tf_size=128, l_rate=l_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])

        #This is the list of callback operations to perform. These occur mid-training and allow us to
        #change parameters or save weights as we go.
        callbacks=[]
        filepath = None

        #Early stopping allows us to end learning after things start to slow down.
        if e_stop:
            earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True)
            callbacks.append(earlystop)

        #The learning rate scheduler allows us to alter the learning rate as the epochs increase.
        if lr_scheduler:
            scheduler = LearningRateScheduler(keras_classifiers.EEGNetScheduler)
            callbacks.append(scheduler)
        #If using a checkpointer, set it up here.
        if check_p:
            #Create the filepath
            filepath = 'E:/PycharmProjects/OpenBCIResearch/NN_Weights/eegnet_best.hdf5'

            #Form the checkpointer and callback list.
            checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                           save_best_only=True, save_weights_only=True)
            callbacks.append(checkpointer)

        #This is the dictionary of details for the fit method.
        fit_dict = {
            'batch_size': 16,
            'epochs': 100,
            'callbacks': callbacks,
            'checkpoint': filepath
        }

        return "eegnet", model, fit_dict

    # An implementation of Deep ConvNet.
    # Available at: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730
    def __deep_convnet(self, data_shape, dropout=0.5, n_classes=2, lr_scheduler=False, check_p=True,
                       e_stop=False, lr_plateau=True, l_rate=0.0001):

        # First, construct the model and compile it.
        model, opt = keras_classifiers.DeepConvNet(nb_classes=n_classes, Chans=data_shape[1], Samples=data_shape[2],
                                                   dropoutRate=dropout, l_rate=l_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])

        # This is the list of callback operations to perform. These occur mid-training and allow us to
        # change parameters or save weights as we go.
        callbacks = []
        filepath = None

        # Early stopping allows us to end learning after things start to slow down.
        if e_stop:
            earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True)
            callbacks.append(earlystop)

        # The learning rate scheduler allows us to alter the learning rate as the epochs increase.
        if lr_scheduler:
            scheduler = LearningRateScheduler(keras_classifiers.EEGNetScheduler)
            callbacks.append(scheduler)

        if lr_plateau:
            plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
            callbacks.append(plat)
        # If using a checkpointer, set it up here.
        if check_p:
            # Create the filepath
            filepath = 'E:/PycharmProjects/OpenBCIResearch/NN_Weights/deepconv_best.hdf5'

            # Form the checkpointer and callback list.
            checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                           save_best_only=True, save_weights_only=True)
            callbacks.append(checkpointer)

        # This is the dictionary of details for the fit method.
        fit_dict = {
            'batch_size': 16,
            'epochs': 100,
            'callbacks': callbacks,
            'checkpoint' : filepath
        }

        return "deep_convnet", model, fit_dict

    # An implementation of Deep ConvNet.
    # Available at: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730
    def __shallow_convnet(self, data_shape, dropout=0.5, n_classes=2, lr_scheduler=False, check_p=True,
                       e_stop=False, lr_plateau=True, l_rate=1e-3):

        # First, construct the model and compile it.
        model, opt = keras_classifiers.ShallowConvNet(nb_classes=n_classes, Chans=data_shape[1], Samples=data_shape[2],
                                                   dropoutRate=dropout, l_rate=l_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])

        # This is the list of callback operations to perform. These occur mid-training and allow us to
        # change parameters or save weights as we go.
        callbacks = []
        filepath = None

        # Early stopping allows us to end learning after things start to slow down.
        if e_stop:
            earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True)
            callbacks.append(earlystop)

        # The learning rate scheduler allows us to alter the learning rate as the epochs increase.
        if lr_scheduler:
            scheduler = LearningRateScheduler(keras_classifiers.EEGNetScheduler)
            callbacks.append(scheduler)

        if lr_plateau:
            plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_delta=1e-4, verbose=1)
            callbacks.append(plat)
        # If using a checkpointer, set it up here.
        if check_p:
            # Create the filepath
            filepath = 'E:/PycharmProjects/OpenBCIResearch/NN_Weights/shallowconv_best.hdf5'

            # Form the checkpointer and callback list.
            checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                           save_best_only=True, save_weights_only=True)
            callbacks.append(checkpointer)

        # This is the dictionary of details for the fit method.
        fit_dict = {
            'batch_size': 16,
            'epochs': 100,
            'callbacks': callbacks,
            'checkpoint' : filepath
        }

        return "shallow_convnet", model, fit_dict

    #Fusion Convolutional Neural Network for Cross-Subject EEG Motor Imagery Classification
    #Available at: https://research.tees.ac.uk/en/publications/fusion-convolutional-neural-network-for-cross-subject-eeg-motor-i
    def __fusion_eegnet(self, data_shape, dropout=0.5, n_classes=2, lr_scheduler=False, check_p=True,
                        e_stop=False, lr_plateau=True, l_rate=1E-3):
        #Form the classifier
        model, opt = keras_classifiers.fusionEEGNet(n_classes=n_classes, chans=data_shape[1], samples=data_shape[2],
                                                    third_axis=data_shape[3], l_rate=l_rate, dropout_rate=dropout)
        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])

        # This is the list of callback operations to perform. These occur mid-training and allow us to
        # change parameters or save weights as we go.
        callbacks = []
        filepath = None

        # Early stopping allows us to end learning after things start to slow down.
        if e_stop:
            earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True)
            callbacks.append(earlystop)

        # The learning rate scheduler allows us to alter the learning rate as the epochs increase.
        if lr_scheduler:
            scheduler = LearningRateScheduler(keras_classifiers.EEGNetScheduler)
            callbacks.append(scheduler)
        # If using a checkpointer, set it up here.
        if lr_plateau:
            plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
            callbacks.append(plat)

        if check_p:
            # Create the filepath
            filepath = 'E:/PycharmProjects/OpenBCIResearch/NN_Weights/eegnet_fusion_best.hdf5'

            # Form the checkpointer and callback list.
            checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                           save_best_only=True, save_weights_only=True)
            callbacks.append(checkpointer)

        # This is the dictionary of details for the fit method.
        fit_dict = {
            'batch_size': 32,
            'epochs': 100,
            'callbacks': callbacks,
            'checkpoint' : filepath
        }

        return "fusion_eegnet", model, fit_dict

    ##------------------------------------------------------------------------------------------------------------------
    ##Testing Methods
    ##------------------------------------------------------------------------------------------------------------------

    #Trains and tests each classifier on each subject individually.
    def run_individual_test(self, sk_test=True, nn_test=True, sk_select=None, nn_select=None, cross_val_times=10):
        # These are internal switches to control which classifiers are used in the train/test internal methods.
        self.sk_test = sk_test
        self.sk_select = sk_select
        self.nn_test = nn_test
        self.nn_select = nn_select

        # See if our classifiers are already generated, and if not, do so.
        if sk_test:
            if not self.sk_class_loaded:
                self.initialise_sklearn_classifiers()
        if nn_test:
            if not self.nn_class_loaded and not self.filter_bank:
                self.initialise_neural_networks()

        # Write our initial text into results.
        self.__print("--INDIVIDUAL TEST--\n")
        self.__print("Parameters:\n")
        self.__print("    sk_test = {sk}, sk_select = {sks}\n".format(sk=sk_test, sks=sk_select))
        self.__print("    nn_test = {nn}, nn_select = {nns}\n".format(nn=nn_test, nns=nn_select))
        self.__print("    cross_val_times = {cvt}\n".format(cvt=cross_val_times))

        for key, sub in self.sub_dict.items():
            self.__print("--Subj No. {n}: \n".format(n=key))
            # Perform the cross validation.
            results = self.__stratified_cross_val(sub['data'], sub['labels'],
                                                  cross_val_times)
            self.__print_average_results(results=results)
        return

    #Trains and tests each classifier on randomised batches of subjects.
    def run_batch_test(self, batch_size, n_times, sk_test=True, nn_test=True, sk_select=None, nn_select=None,
                       test_split=0.2, split_subject=False, cross_val_times=10, avg=True):
        #Check parameters.
        if n_times < 1:
            raise Exception("Error: Attribute 'n_times' must be greater than or equal to 1.")
        if batch_size <= 1:
            raise Exception("Error: Attribute 'batch_size' must be greater than 1.")
        if test_split < 0.1 or test_split > 0.5:
            raise Exception("Error: Attribute 'test_split' must be a value between 0.1 and 0.5 inclusive.")

        #These are internal switches to control which classifiers are used in the train/test internal methods.
        self.sk_test = sk_test
        self.sk_select = sk_select
        self.nn_test = nn_test
        self.nn_select = nn_select

        # See if our classifiers are already generated, and if not, do so.
        if sk_test:
            if not self.sk_class_loaded:
                self.initialise_sklearn_classifiers()
        if nn_test:
            if not self.nn_class_loaded and not self.filter_bank:
                self.initialise_neural_networks()

        if batch_size > len(self.sub_dict.keys()):
            batch_size = len(self.sub_dict.keys())

        #Write our initial text into results.
        self.__print("--BATCH TEST--\n")
        self.__print("Parameters:\n")
        self.__print("    batch_size = {batch}\n".format(batch=batch_size))
        self.__print("    n_times = {n}\n".format(n=n_times))
        self.__print("    sk_test = {sk}, sk_select = {sks}\n".format(sk=sk_test, sks=sk_select))
        self.__print("    nn_test = {nn}, nn_select = {nns}\n".format(nn=nn_test, nns=nn_select))
        self.__print("    train_test_split = {tts}, split_subjects = {ss}\n".format(tts=train_test_split,
                                                                                              ss=split_subject))
        self.__print("    cross_val_times = {cvt}\n".format(cvt=cross_val_times))
        results = []
        for x in range(0, n_times):
            if split_subject:
                results.append(self.__split_subject_train_test_classifiers(batch_size=batch_size,
                                                                           cross_val_times=cross_val_times,
                                                                           test_split=test_split))
            else:
                results.append(self.__train_test_classifiers(batch_size=batch_size, cross_val_times=cross_val_times))

        count = 1
        for result in results:
            self.__print("--Batch No. {n}: \n".format(n=count))
            if avg:
                self.__print_average_results(result)
            else:
                self.__print_non_average_results(result)
            self.__print("\n")
            count = count + 1
        return

    #Trains and tests each classifier on incrementally growing randomised batches of subjects.
    def run_increment_batch_test(self, batch_size, incr_value, max_batch_size=None, sk_test=True, nn_test=True,
                                 sk_select=None, nn_select=None, test_split=0.2, split_subject=False,
                                 n_times=10):
        # Check parameters.
        if n_times < 3:
            raise Exception("Error: Attribute 'cross_val_times' must be greater than or equal to 3.")
        if batch_size <= 1:
            raise Exception("Error: Attribute 'batch_size' must be greater than 1.")
        if max_batch_size < batch_size:
            raise Exception("Error: Attribute 'max_batch_size' must be greater than attribute 'batch_size'.")
        if test_split < 0.1 or test_split > 0.5:
            raise Exception("Error: Attribute 'test_split' must be a value between 0.1 and 0.5 inclusive.")
        if max_batch_size > len(self.sub_dict.keys()):
            max_batch_size = len(self.sub_dict.keys())
            self.__print(
                "\nWarning: parameter 'max_batch_size' has been decreased to the total subject count {num}\n".format(
                    num=max_batch_size))

        # These are internal switches to control which classifiers are used in the train/test internal methods.
        self.sk_test = sk_test
        self.sk_select = sk_select
        self.nn_test = nn_test
        self.nn_select = nn_select

        # See if our classifiers are already generated, and if not, do so.
        if sk_test:
            if not self.sk_class_loaded:
                self.initialise_sklearn_classifiers()
        if nn_test:
            if not self.nn_class_loaded and not self.filter_bank:
                self.initialise_neural_networks()

        # Write our initial text into results.
        self.__print("--INCREMENTAL BATCH TEST--\n")
        self.__print("Parameters:\n")
        self.__print("    batch_size = {batch}\n".format(batch=batch_size))
        self.__print("    incr_value = {inc}\n".format(inc=incr_value))
        self.__print("    max_batch_size = {mbs}\n".format(mbs=max_batch_size))
        self.__print("    sk_test = {sk}, sk_select = {sks}\n".format(sk=sk_test, sks=sk_select))
        self.__print("    nn_test = {nn}, nn_select = {nns}\n".format(nn=nn_test, nns=nn_select))
        self.__print("    train_test_split = {tts}, split_subjects = {ss}\n".format(tts=train_test_split,
                                                                                    ss=split_subject))
        self.__print("    cross_val_times = {cvt}\n".format(cvt=n_times))

        #Now, we perform the test and increment as we go.
        results = {}
        cur_batch_size = batch_size
        while cur_batch_size < max_batch_size:
            if split_subject:
                result = self.__split_subject_train_test_classifiers(batch_size=cur_batch_size,
                                                                     cross_val_times=n_times,
                                                                     test_split=test_split)
            else:
                result = self.__train_test_classifiers(batch_size=cur_batch_size, cross_val_times=n_times)
            results[cur_batch_size] = result
            cur_batch_size = cur_batch_size + incr_value

        for b_size, result in results.items():
            self.__print("--Batch Size: {n}: \n".format(n=b_size))
            self.__print_average_results(result)
            self.__print("\n")
        return

    #Performs a leave-one-out test on the specified data range.
    def run_LOO_test(self, sk_test=True, nn_test=True, sk_select=None, nn_select=None):
        # These are internal switches to control which classifiers are used in the train/test internal methods.
        self.sk_test = sk_test
        self.sk_select = sk_select
        self.nn_test = nn_test
        self.nn_select = nn_select

        # See if our classifiers are already generated, and if not, do so.
        if sk_test:
            if not self.sk_class_loaded:
                self.initialise_sklearn_classifiers()
        if nn_test:
            if not self.nn_class_loaded and not self.filter_bank:
                self.initialise_neural_networks()

        # Write our initial text into results.
        self.__print("--LEAVE ONE OUT TEST--\n")
        self.__print("Parameters:\n")
        self.__print("    sk_test = {sk}, sk_select = {sks}\n".format(sk=sk_test, sks=sk_select))
        self.__print("    nn_test = {nn}, nn_select = {nns}\n".format(nn=nn_test, nns=nn_select))

        results = {}
        subjects = list(self.sub_dict.keys())
        data = None
        labels = None
        for sub1 in subjects:
            f_it = True
            for sub2 in subjects:
                if sub2 == sub1:
                    continue
                #For our first non-sub1 iteration, we need to assign data and labels.
                if f_it:
                    data = self.sub_dict[sub2]['data']
                    labels = self.sub_dict[sub2]['labels']
                    f_it = False
                    continue

                #And we contatenate.
                data = np.concatenate(
                    (data, self.sub_dict[sub2]['data']),
                    axis=0)
                labels = np.concatenate(
                    (labels, self.sub_dict[sub2]['labels']),
                    axis=0)

            test_data, test_labels = self.sub_dict[sub1]['data'], self.sub_dict[sub1]['labels']
            data, labels = shuffle(data, labels)
            test_data, test_labels = shuffle(test_data, test_labels)

            #Now, we train on our data, and test on the subject.
            self.__train(data, labels)
            results[sub1] = self.__test(test_data, test_labels)

        #Now we print.
        for sub, result in results.items():
            self.__print("--Subject Left Out: {n}: \n".format(n=sub))
            if self.avg:
                self.__print_average_results(result)
            else:
                self.__print_results(result)
            self.__print("\n")

        return

    ##------------------------------------------------------------------------------------------------------------------
    ##Testing Tools
    ##------------------------------------------------------------------------------------------------------------------
    def __clear(self):
        self.nn_dict.clear()
        backend.clear_session()
        gc.collect()
        return

    def __train(self, data, labels):
        if self.sk_test:
            for name, classifier in self.sk_dict.items():
                if self.sk_select is None or name in self.sk_select:
                    classifier.fit(data, labels)
        if self.nn_test:
            #First, remember that Keras models, when called with fit(), do not overwrite previous weights,
            #but rather continue to train from where they were left. So, we re-initialise the networks now.
            self.__clear()
            self.initialise_neural_networks()


            #For each NN, split into
            for name, network in self.nn_dict.items():
                if self.nn_select is None or name in self.nn_select:
                    # Reshape the data, split into train/validation sets and one-hot encode the labels.
                    newdata = gen_tools.reshape_3to4(data)
                    newdata = newdata * 1000  # This scaling is apparently needed according to EEGNet source.
                    T_data, V_data, t_labels, v_labels = train_test_split(newdata, labels, test_size=0.125, shuffle=True)
                    t_labels = to_categorical(t_labels, 2)
                    v_labels = to_categorical(v_labels, 2)

                    #First, we fit.
                    network[0].fit(T_data, t_labels, batch_size=network[1]['batch_size'],
                               epochs=network[1]['epochs'], callbacks=network[1]['callbacks'],
                                   validation_data=(V_data, v_labels), class_weight={0:1, 1:1})
                    #Now, if we have a checkpointer, we load the best weights.
                    if network[1]['checkpoint'] is not None:
                        network[0].load_weights(network[1]['checkpoint'])
                        #Now, we delete our weights as we don't want them re-used.
                        if path.isfile(network[1]['checkpoint']):
                            remove(network[1]['checkpoint'])
                        else:
                            raise Exception("ERROR: Network {net} weights at path {path} cannot be deleted.".format(
                                net=name, path=network[1]['checkpoint']))
                    else:
                        print("WARNING: No checkpoint has been set. NN Model will not use best epoch.")

        return

    def __test(self, data, labels):
        #Initialise the result dictionary.
        #Will be in form 'Classifier Name': 'Single Trial Result'.
        #Where 'Single Trial Result' is itself a dictionary.
        results = {}

        #Perform single trial test for sk learn format models.
        if self.sk_test:
            for name, classifier in self.sk_dict.items():
                if self.sk_select is None or name in self.sk_select: #Thank the lord for short-circuit evaluation...
                    #Make the predictions.
                    preds = classifier.predict(data)
                    results[name] = {}

                    prefix = 'test_'
                    #Evaluate based on metrics.
                    for metric in self.result_metrics.keys():
                        if metric == 'Accuracy':
                            results[name][prefix + metric] = accuracy_score(labels, preds)
                        elif metric == 'Recall':
                            results[name][prefix + metric] = recall_score(labels, preds)
                        elif metric == 'Precision':
                            results[name][prefix + metric] = precision_score(labels, preds)
                        elif metric == 'F1_Score':
                            results[name][prefix + metric] = f1_score(labels, preds)
                        elif metric == 'ROC_AUC':
                            results[name][prefix + metric] = roc_auc_score(labels, preds)

        #Perform single trial test for neural network models.
        if self.nn_test:
            #Got to reshape for our neural network approach.
            newdata = gen_tools.reshape_3to4(data)
            newdata = newdata * 1000  # This scaling is apparently needed according to EEGNet source.

            for name, network in self.nn_dict.items():
                if self.nn_select is None or name in self.nn_select:  # Thank the lord for short-circuit evaluation...
                    # Make the predictions.
                    probs = network[0].predict(newdata)

                    #Using Softmax as the activation gives us a probability for each class, so
                    #we need to turn them into the most likely prediction. i.e. make it a non-categorical result.
                    preds = probs.argmax(axis=-1)
                    results[name] = {}

                    prefix = 'test_'
                    # Evaluate based on metrics.
                    for metric in self.result_metrics.keys():
                        if metric == 'Accuracy':
                            results[name][prefix + metric] = accuracy_score(labels, preds)
                        elif metric == 'Recall':
                            results[name][prefix + metric] = recall_score(labels, preds)
                        elif metric == 'Precision':
                            results[name][prefix + metric] = precision_score(labels, preds)
                        elif metric == 'F1_Score':
                            results[name][prefix + metric] = f1_score(labels, preds)
                        elif metric == 'ROC_AUC':
                            results[name][prefix + metric] = roc_auc_score(labels, preds)
                    print(results[name])
        return results

    def __stratified_cross_val(self, data, labels, cv=5):
        #Initialise the result dictionary.
        #will be in form 'Classifier Name' : 'Cross Validation Results'
        results = {}

        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        #perform stratified cross validation for sk learn format models.
        if self.sk_test:
            for name, classifier in self.sk_dict.items():
                if self.sk_select is None or name in self.sk_select: #Thank the lord for short-circuit evaluation...
                    scores = cross_validate(classifier, data, labels, scoring=self.result_metrics,
                         cv=kfold, return_train_score=True)
                    results[name] = scores

        # Perform stratified cross validation for neural network models.
        if self.nn_test:
            # Not Implemented
            not_imp = None

        return results

    def __train_test_classifiers(self, batch_size, cross_val_times):
        # Create our randomised set of subjects
        subjects = list(self.sub_dict.keys())
        random.shuffle(subjects)
        subjects = subjects[:batch_size]

        # Combine and randomise the data from subjects.
        cur_sub = subjects.pop()
        data = self.sub_dict[cur_sub]['data']
        labels = self.sub_dict[cur_sub]['labels']
        while len(subjects) > 0:
            cur_sub = subjects.pop()
            data = np.concatenate(
                (data, self.sub_dict[cur_sub]['data']),
                axis=0)
            labels = np.concatenate(
                (labels, self.sub_dict[cur_sub]['labels']),
                axis=0)
        data, labels = shuffle(data, labels)

        #Perform the cross validation.
        results = self.__stratified_cross_val(data, labels, cross_val_times)
        return results

    def __split_subject_train_test_classifiers(self, batch_size, cross_val_times, test_split):
        # Create our randomised set of subjects
        subjects = list(self.sub_dict.keys())
        random.shuffle(subjects)
        subjects = subjects[:batch_size]
        subject_set = set(subjects)

        # Get the nearest whole number for the train test split.
        # i.e. if there are 10 subjects, 0.2 train_test_split, then we get 2 subjects for testing.
        test_count = round(len(subjects) * test_split)

        results = []
        cur_index = 0
        tested_set = set([])
        for x in range(0, cross_val_times):
            #We find our test and train sets by some basic set operations.
            test_set = set(random.sample((subject_set - tested_set), k=test_count))
            train_set = subject_set - test_set

            # Now, if our tested set has grown too large for the next loop, we reset it, otherwise
            # we simply combine it with our current test set.
            if len(subject_set - (tested_set.union(test_set))) < test_count:
                tested_set = set([])
            else:
                tested_set = tested_set.union(test_set)

            #Grab our test data by using the test set
            cur_sub = test_set.pop()
            data_test = self.sub_dict[cur_sub]['data']
            labels_test = self.sub_dict[cur_sub]['labels']
            while len(test_set) > 0:
                cur_sub = test_set.pop()
                data_test = np.concatenate(
                    (data_test, self.sub_dict[cur_sub]['data']),
                    axis=0)
                labels_test = np.concatenate(
                    (labels_test, self.sub_dict[cur_sub]['labels']),
                    axis=0)

            # Fill training data with the rest
            cur_sub = train_set.pop()
            data_train = self.sub_dict[cur_sub]['data']
            labels_train = self.sub_dict[cur_sub]['labels']
            while len(train_set) > 0:
                cur_sub = train_set.pop()
                data_train = np.concatenate(
                    (data_train, self.sub_dict[cur_sub]['data']),
                    axis=0)
                labels_train = np.concatenate(
                    (labels_train, self.sub_dict[cur_sub]['labels']),
                    axis=0)

            # Randomise them both. Don't want them ordered.
            data_test, labels_test = shuffle(data_test, labels_test)
            data_train, labels_train = shuffle(data_train, labels_train)

            self.__train(data_train, labels_train)
            results.append(self.__test(data_test, labels_test))

        #Now, we need to change results from the form of a list(dict(dict()))
        #to the form of dict(dict()) to match the cross validate function and make it
        #easier to read.
        final_results = results[0]
        count = 1
        #Here, we iterate through the key pairs, adding the values and finding the average.
        for name, metrics in final_results.items():
            for key in metrics.keys():
                res_list = []
                for res in results:
                    res_list.append(res[name][key])
                final_results[name][key] = res_list
        return final_results