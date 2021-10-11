from ClassifierTester import ClassifierTester
import random

#The sets of stimuli/operations to test.
combinations = [('hf', 'movement'), ('lr', 'movement')]
rand = random.randint(1, 999999)

#Leave-One-Out test on OpenBCI data.
for combo in combinations:
    print("\nIterating for Combination: {c1}-{c2}\n\n".format(c1=combo[0], c2=combo[1]))
    # Form our path/filename. Here, we're saving somewhere different to the default to make them easy to find.
    fname = 'loo_{stim}_{type}_openbci'.format(stim=combo[0], type=combo[1])
    fpath = 'E:/PycharmProjects/OpenBCIResearch/CLassifierTesterResults/Batch/NN'

    # For ML
    test = ClassifierTester(subj_range=[1, 6], data_source='live-movement', stim_select=combo[0], stim_type=combo[1],
                            result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], tmin=-1, tmax=4, notch=None,
                            f_name=fname + '_ml', f_path=fpath, random_state=rand, p_select='genetic', p_select_frac=1,
                            filter_bounds=(8., 35.), live_layout='m_cortex')
    test.run_LOO_test(sk_test=True)
    del test

    # For NNs
    test = ClassifierTester(subj_range=[1, 6], data_source='live-movement', stim_select=combo[0], stim_type=combo[1],
                            result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], tmin=0, tmax=4, notch=50,
                            f_name=fname + '_nn', f_path=fpath, random_state=rand, p_select=None, filter_bounds=(2., 60.),
                            live_layout='m_cortex')
    test.run_LOO_test(nn_test=True)
    del test
