from ClassifierTester import ClassifierTester
import random

#The sets of stimuli/operations to test.
combinations = [('hf', 'movement'), ('lr', 'movement')]
rand = random.randint(1, 999999)

#Leave-One-Out test on OpenBCI data.
#Note, the actual LOO test had some issues, so the batch test with a test split of 0.2 and split subjects fixes this.
for combo in combinations:
    print("\nIterating for Combination: {c1}-{c2}\n\n".format(c1=combo[0], c2=combo[1]))
    # Form our path/filename. Here, we're saving somewhere different to the default to make them easy to find.
    fname = 'loo_{stim}_{type}_openbci_new'.format(stim=combo[0], type=combo[1])
    fpath = 'E:/PycharmProjects/OpenBCIResearch/CLassifierTesterResults/LeaveOneOut/'

    # For ML
    ml_path = fpath + 'ML'
    ml_name = fname + '_ml'
    test = ClassifierTester(subj_range=[1, 6], data_source='live-movement', stim_select=combo[0], stim_type=combo[1],
                            result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], tmin=-1, tmax=4, notch=None,
                            f_name=ml_name, f_path=ml_path, random_state=rand, p_select='genetic', p_select_frac=1,
                            filter_bounds=(8., 35.), live_layout='m_cortex')
    #test.run_LOO_test(sk_test=True, nn_test=False)
    test.run_batch_test(batch_size=5, n_times=1, sk_test=True, test_split=0.2, cross_val_times=5, nn_test=False,
                        split_subject=True, avg=False)
    del test

    # For fb
    fb_name = fname + '_ml_fb'
    test = ClassifierTester(subj_range=[1, 6], data_source='live-movement', stim_select=combo[0], stim_type=combo[1],
                            result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], tmin=-1, tmax=4, notch=None,
                            f_name=fb_name, f_path=ml_path, random_state=rand, p_select='genetic', p_select_frac=1,
                            live_layout='m_cortex', filter_bank=True)
    #test.run_LOO_test(sk_test=True, nn_test=False)
    test.run_batch_test(batch_size=5, n_times=1, sk_test=True, test_split=0.2, cross_val_times=5, nn_test=False,
                        split_subject=True, avg=False)
    del test

    #For NNs
    nn_path = fpath + 'NN'
    nn_name = fname + '_nn'
    test = ClassifierTester(subj_range=[1, 6], data_source='live-movement', stim_select=combo[0], stim_type=combo[1],
                            result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], tmin=0, tmax=4, notch=50,
                            f_name=nn_name, f_path=nn_path, random_state=rand, p_select=None, filter_bounds=(2., 60.),
                            live_layout='m_cortex')
    test.run_batch_test(batch_size=5, n_times=5, nn_test=True, test_split=0.2, cross_val_times=5, sk_test=False,
                        split_subject=True)
    del test
