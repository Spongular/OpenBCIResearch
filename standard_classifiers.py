# This file contains the implementation for all of the classification methods.
import gen_tools
import numpy as np
import matplotlib.pyplot as plt
from mne import pick_types, Epochs, events_from_annotations, concatenate_raws
from mne.channels import make_standard_montage
from mne.decoding import CSP, Scaler, UnsupervisedSpatialFilter, Vectorizer
from mne.datasets.eegbci import eegbci
from mne.io import read_raw_edf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split, GridSearchCV
#from sklearn.model_selection import HalvingGridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from time import time

# 0 - Internal Methods

def __perform_gridsearch(classifier, parameters, data, labels, n_jobs, verbose=0, cross_val=3):
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
    return

def __perform_halving_gridsearch(classifier, parameters, data, labels, n_jobs, verbose=0, cross_val=3,
                                 halv_factor=3):
    # Here, we make use of the CVGridsearch method to check the
    # various combinations of parameters for the best result.
    print("Performing HalvingGridSearchCV to find optimal parameter set...")
    t0 = time()
    h_grid_search = HalvingGridSearchCV(classifier, parameters, n_jobs=n_jobs, verbose=verbose,
                                      cv=cross_val, factor=halv_factor)
    h_grid_search.fit(data, labels)
    print("HalvingGridSearchCV completed in %0.3fs" % (time() - t0))

    # And print out our results.
    print("Displaying Results...")
    print("Best score: %0.3f" % h_grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = h_grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return

def __perform_randomsearch(classifier, parameters, data, labels, n_jobs, verbose=0, cross_val=3,
                           iterations=10):
    # Here, we make use of the CVGridsearch method to check the
    # various combinations of parameters for the best result.
    print("Performing RandomizedSearchCV to find optimal parameter set...")
    t0 = time()
    random_search = RandomizedSearchCV(classifier, parameters, n_jobs=n_jobs, verbose=verbose,
                                     cv=cross_val, n_iter=iterations)
    random_search.fit(data, labels)
    print("RandomizedSearchCV completed in %0.3fs" % (time() - t0))

    # And print out our results.
    print("Displaying Results...")
    print("Best score: %0.3f" % random_search.best_score_)
    print("Best parameters set:")
    best_parameters = random_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return


def __perform_halving_randomsearch(classifier, parameters, data, labels, n_jobs, verbose=0, cross_val=3,
                                   iterations=10, halv_factor=3):
    # Here, we make use of the CVGridsearch method to check the
    # various combinations of parameters for the best result.
    print("Performing HalvingRandomSearchCV to find optimal parameter set...")
    t0 = time()
    random_search = HalvingRandomSearchCV(classifier, parameters, n_jobs=n_jobs, verbose=verbose,
                                          cv=cross_val, n_iter=iterations, factor=halv_factor)
    random_search.fit(data, labels)
    print("HalvingRandomSearchCV completed in %0.3fs" % (time() - t0))

    # And print out our results.
    print("Displaying Results...")
    print("Best score: %0.3f" % random_search.best_score_)
    print("Best parameters set:")
    best_parameters = random_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return

# 1 - Motor Imagery

def csp_lda(raw, tmin, tmax, pick_list=[], plot_csp=False, n_logspace=50, n_jobs=1, max_csp_components=20):

    #Grab the data, labels and epochs
    data, labels, epochs = gen_tools.epoch_data(raw, tmin=tmin, tmax=tmax, pick_list=pick_list)

    # Assemble classifiers
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn pipeline with cross_val_score function
    # Run it on all the data because we're lazy.
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, data, labels, cv=cv, n_jobs=1)

    # Here, we are cross-validating the LDA and CSP pipeline to find
    # the number of components for CSP that provides the best result.
    # The loop determines the CSP values, from 2 through to 10 initially.
    # The default is 4.
    # The values below are necessary for keeping track.
    best_n = 0
    best_score = 0
    std_dev = 0
    for n in range(2, max_csp_components):
        # Here, we change the N value of the csp.
        csp.n_components = n
        cross_score = cross_val_score(clf, data, labels, cv=cv, n_jobs=n_jobs)
        # Change our values if they are better.
        if np.mean(cross_score) > best_score:
            best_score = np.mean(cross_score)
            std_dev = np.std(cross_score)
            best_n = n
        print("LDA Mid-CrossVal acc: %f, std: %f, n_comp = %f" % (best_score, std_dev, best_n))

    #This is for if we want to plot the CSP to view.
    #For some reason, we can't plot the patterns with only a select
    #few picks, so we limit it to only allow it when no picks are made.
    if plot_csp and len(pick_list) == 0:
        csp.fit_transform(data, labels)
        csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
        plt.show(block=True)
    elif plot_csp:
        print("Cannot Plot CSP with reduced channel list")

    print("LDA Cross Validation accuracy: %f with std deviation of %f" % (best_score, std_dev))
    print("CSP n_components is %f" % (best_n))
    return

def csp_svm(raw, tmin, tmax, pick_list=[], plot_csp=False, n_logspace=50, n_jobs=1, max_csp_components=20):
    # If our function parameters are invalid, set them right.
    if max_csp_components < 2: #Components cannot be less than two.
        max_csp_components = 20
    if n_jobs > 4 or n_jobs < -1: #-1 means all cores, greater than four may lead to issues.
        n_jobs = 1
    if n_logspace < 1: #n_logspace needs to be positive.
        n_logspace = 50

    # Grab the data, labels and epochs
    data, labels, epochs = gen_tools.epoch_data(raw, tmin=tmin, tmax=tmax, pick_list=pick_list)

    # Assemble classifiers
    svc = svm.SVC(kernel='linear', C=1, random_state=42)
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn pipeline with cross_val_score function
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    clf = Pipeline([('CSP', csp), ('SVM', svc)])

    #Here, we are cross-validating the SVM and CSP to find the correct
    #C value and number of components respectively. The outer loop determines
    #the CSP values, from 2 through to 10 initially. The default is 4.
    #The values below are necessary for keeping track.
    best_n = 0
    c_vals = np.logspace(-10, 10, 10, num=n_logspace)
    best_c = 0
    best_score = 0
    std_dev = 0
    for n in range(2, max_csp_components):
        #Here, we change the N value of the csp.
        csp.n_components = n

        #Now, we want to iterate through potential C values for the SVM.
        #The C value changes the tolerance for outliers in the hyperplane.
        #A higher C value means lower tolerance. Tuning C allows us to find the
        #best compromise that allows for the highest accuracy.
        for c in c_vals:
            #Set C and get the score.
            svc.C = c
            cross_score = cross_val_score(clf, data, labels, cv=cv, n_jobs=n_jobs)
            #Change our values if they are better.
            if np.mean(cross_score) > best_score:
                best_score = np.mean(cross_score)
                best_c = c
                std_dev = np.std(cross_score)
                best_n = n
        print("SVM Mid-CrossVal acc: %f, std: %f, c = %f, n_comp = %f" % (best_score, std_dev, best_c, best_n))

    # This is for if we want to plot the CSP to view.
    # For some reason, we can't plot the patterns with only a select
    # few picks, so we limit it to only allow it when no picks are made.
    if plot_csp and len(pick_list) == 0:
        csp.fit_transform(data, labels)
        csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
        plt.show(block=True)
    elif plot_csp:
        print("Cannot Plot CSP with reduced channel list")

    print("SVM Cross Validation accuracy: %f with std deviation of %f" % (best_score, std_dev))
    print("SVM C value is %f, and CSP n_components is %f" %(best_c, best_n))
    return


def csp_knn(raw, tmin, tmax, pick_list=[], plot_csp=False, max_n_neighbors=4, n_jobs=1, max_csp_components=20):
    print("CSP-KNN test for Motor Imagery beginning...")
    print("Checking Parameters...")
    # If our function parameters are invalid, set them right.
    if max_csp_components < 2:  # Components cannot be less than two.
        print("Parameter 'max_csp_components' is below 2, setting to default.")
        max_csp_components = 20
    if n_jobs > 4 or n_jobs < -1:  # -1 means all cores, greater than four may lead to issues.
        print("Parameter 'n_jobs' is outside of range -1,4, setting to default.")
        n_jobs = 1
    if max_n_neighbors < 2:  # n_neighbors needs to be positive.
        print("Parameter 'n_neighbors' is below 2, setting to default.")
        max_n_neighbors = 4
    print("Parameters checked.")

    # Grab the data, labels and epochs
    data, labels, epochs = gen_tools.epoch_data(raw, tmin=tmin, tmax=tmax, pick_list=pick_list)

    print("Assembling classification pipeline and parameter set...")
    # Assemble classifiers
    knn = KNeighborsClassifier()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
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

    #Here, we make use of the CVGridsearch method to check the
    #various combinations of parameters for the best result.
    print("Performing CVGridSearch to find optimal parameter set...")
    t0 = time()
    grid_search = GridSearchCV(clf, parameters, n_jobs=n_jobs, verbose=0)
    grid_search.fit(data, labels)
    print("CVGridSearch completed in %0.3fs" % (time() - t0))

    # This is for if we want to plot the CSP to view.
    # For some reason, we can't plot the patterns with only a select
    # few picks, so we limit it to only allow it when no picks are made.
    if plot_csp and len(pick_list) == 0:
        print("Plotting CSP...")
        #Grab our best parameters
        param_set = grid_search.best_estimator_.get_params()
        csp.n_components = param_set['CSP__n_components']
        csp.cov_est = param_set['CSP__cov_est']
        csp.norm_trace = param_set['CSP__norm_trace']
        #And fit then plot the csp.
        csp.fit_transform(data, labels)
        csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
        plt.show(block=True)
    elif plot_csp:
        print("Cannot Plot CSP with reduced channel list")

    #And print out our results.
    print("Displaying Results...")
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return

def pca_lda(raw, tmin, tmax, pick_list=[], n_jobs=1, pca_n_components=32):
    print("PCA-LDA test for Motor Imagery beginning...")
    print("Checking Parameters...")
    # If our function parameters are invalid, set them right.
    if n_jobs > 4 or n_jobs < -1:  # -1 means all cores, greater than four may lead to issues.
        print("Parameter 'n_jobs' is outside of range -1,4, setting to default.")
        n_jobs = 1
    if pca_n_components < 2:  # -1 means all cores, greater than four may lead to issues.
        print("Parameter 'pca_n_components' is below 2, setting to default.")
        pca_n_components = 32
    print("Parameters checked.")

    # Grab the data, labels and epochs
    data, labels, epochs = gen_tools.epoch_data(raw, tmin=tmin, tmax=tmax, pick_list=pick_list)

    print("Assembling classification pipeline and parameter set...")
    # Assemble classifiers
    vect = Vectorizer()  # Sklearn requires 2d arrays, so we vectorise them.
    scl = StandardScaler()  # PCA requires standardised data.
    pca = PCA(n_components=pca_n_components)  # This reduces the dimensionality of the data.
    kbest = SelectKBest()  # This selects the best features from PCA.
    lda = LinearDiscriminantAnalysis()  # This classifies it all.
    # create the pipeline
    clf = Pipeline([('VECT', vect), ('SCL', scl), ('PCA', pca), ('KBEST', kbest), ('LDA', lda)])
    # Form the parameter dictionary
    parameters = {
        'LDA__solver': ('svd', 'lsqr', 'eigen'),
        'PCA__whiten': (True, False),
        'KBEST__k': np.linspace(2, pca_n_components, 5).astype(int).tolist(),
        'KBEST__score_func': (f_classif, mutual_info_classif, f_regression, mutual_info_regression)
    }

    # Perform our gridsearch on the classifier for the parameter set.
    __perform_gridsearch(classifier=clf, parameters=parameters, data=data, labels=labels, n_jobs=n_jobs)

    return

def pca_svm(raw, tmin, tmax, pick_list=[], n_logspace=10, n_jobs=1, pca_n_components=32):
    print("PCA-SVM test for Motor Imagery beginning...")
    print("Checking Parameters...")
    # If our function parameters are invalid, set them right.
    if n_jobs > 4 or n_jobs < -1:  # -1 means all cores, greater than four may lead to issues.
        print("Parameter 'n_jobs' is outside of range -1,4, setting to default.")
        n_jobs = 1
    if n_logspace < 2:  # n_neighbors needs to be positive.
        print("Parameter 'n_logspace' is below 2, setting to default.")
        n_logspace = 10
    if pca_n_components < 2:  # -1 means all cores, greater than four may lead to issues.
        print("Parameter 'pca_n_components' is below 2, setting to default.")
        pca_n_components = 32
    print("Parameters checked.")

    # Grab the data, labels and epochs
    data, labels, epochs = gen_tools.epoch_data(raw, tmin=tmin, tmax=tmax, pick_list=pick_list)

    print("Assembling classification pipeline and parameter set...")
    # Assemble classifiers
    vect = Vectorizer()  # Sklearn requires 2d arrays, so we vectorise them.
    scl = StandardScaler()  # PCA requires standardised data.
    pca = PCA(n_components=pca_n_components)  # This reduces the dimensionality of the data.
    kbest = SelectKBest()  # This selects the best features from PCA.
    svc = svm.SVC()  # This classifies it all.
    # create the pipeline
    clf = Pipeline([('VECT', vect), ('SCL', scl),  ('PCA', pca), ('KBEST', kbest), ('SVC', svc)])
    # Form the parameter dictionary, ('KBEST', kbest)
    parameters = {
        'SVC__C': np.logspace(start=1, stop=10, num=n_logspace, base=10).tolist(),
        #'SVC__kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
        'PCA__whiten': (True, False),
        'KBEST__k': np.linspace(2, pca_n_components, 5).astype(int).tolist(),
        'KBEST__score_func': (f_classif, mutual_info_classif, f_regression, mutual_info_regression)
    }

    # Perform our gridsearch on the classifier for the parameter set.
    __perform_gridsearch(classifier=clf, parameters=parameters, data=data, labels=labels, n_jobs=n_jobs)

    return

def pca_knn(raw, tmin, tmax, pick_list=[], max_n_neighbors=4, n_jobs=1, pca_n_components=32):
    print("PCA-KNN test for Motor Imagery beginning...")
    print("Checking Parameters...")
    # If our function parameters are invalid, set them right.
    if n_jobs > 4 or n_jobs < -1:  # -1 means all cores, greater than four may lead to issues.
        print("Parameter 'n_jobs' is outside of range -1,4, setting to default.")
        n_jobs = 1
    if max_n_neighbors < 2:  # n_neighbors needs to be positive.
        print("Parameter 'n_neighbors' is below 2, setting to default.")
        max_n_neighbors = 4
    if pca_n_components < 2:  # -1 means all cores, greater than four may lead to issues.
        print("Parameter 'pca_n_components' is below 2, setting to default.")
        pca_n_components = 32
    print("Parameters checked.")

    # Grab the data, labels and epochs
    data, labels, epochs = gen_tools.epoch_data(raw, tmin=tmin, tmax=tmax, pick_list=pick_list)

    print("Assembling classification pipeline and parameter set...")
    # Assemble classifiers
    vect = Vectorizer() #Sklearn requires 2d arrays, so we vectorise them.
    scl = StandardScaler() #PCA requires standardised data.
    pca = PCA(n_components=pca_n_components)#This reduces the dimensionality of the data.
    kbest = SelectKBest() #This selects the best features from PCA.
    knn = KNeighborsClassifier() #This classifies it all.
    # create the pipeline
    clf = Pipeline([('VECT', vect), ('SCL', scl),  ('PCA', pca), ('KBEST', kbest), ('KNN', knn)])
    # Form the parameter dictionary, ('KBEST', kbest)
    parameters = {
        'KNN__n_neighbors': range(2, max_n_neighbors + 1),
        'KNN__weights': ('uniform', 'distance'),
        'KNN__algorithm': ('ball_tree', 'kd_tree', 'brute'),
        'PCA__whiten': (True, False),
        'KBEST__k': np.linspace(2, pca_n_components, 5).astype(int).tolist(),
        'KBEST__score_func': (f_classif, mutual_info_classif, f_regression, mutual_info_regression)
    }

    #Perform our gridsearch on the classifier for the parameter set.
    __perform_gridsearch(classifier=clf, parameters=parameters, data=data, labels=labels, n_jobs=n_jobs)

    return

