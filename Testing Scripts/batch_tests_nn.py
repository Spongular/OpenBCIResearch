from ClassifierTester import ClassifierTester
import random

#The sets of stimuli/operations to test.
# combinations = [('hf', 'imaginary'), ('hf', 'movement'),
#                 ('lr', 'imaginary'), ('lr', 'movement')]
combinations = [('lr', 'movement')]

rand = random.randint(1, 999999)

#For Machine Learning---------------------------------------------------------------------------------------------------
#Full 64 Channels
for combo in combinations:
    print("\nIterating for Combination: {c1}-{c2}\n\n".format(c1=combo[0], c2=combo[1]))
    # Form our path/filename. Here, we're saving somewhere different to the default to make them easy to find.
    fname = 'batch_{stim}_{type}_64ch_nn-test'.format(stim=combo[0], type=combo[1])
    fpath = 'E:/PycharmProjects/OpenBCIResearch/CLassifierTesterResults/Batch/NN'

    # Form our testing class and run it.
    test = ClassifierTester(subj_range=[1, 110], data_source='physionet', stim_select=combo[0], stim_type=combo[1],
                            result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], tmin=0, tmax=4, notch=60,
                            f_name=fname, f_path=fpath, random_state=rand, p_select=None, filter_bounds=(2., 60.))
    test.run_increment_batch_test(batch_size=10, incr_value=90, max_batch_size=109, n_times=3, sk_test=False,
                                  nn_test=True, split_subject=True, nn_select=['eegnet'], test_split=0.2)
    del test

#Headband Config
for combo in combinations:
    print("\nIterating for Combination: {c1}-{c2}\n\n".format(c1=combo[0], c2=combo[1]))
    # Form our path/filename. Here, we're saving somewhere different to the default to make them easy to find.
    fname = 'batch_{stim}_{type}_headband_nn'.format(stim=combo[0], type=combo[1])
    fpath = 'E:/PycharmProjects/OpenBCIResearch/CLassifierTesterResults/Batch/NN'

    # Form our testing class and run it.
    test = ClassifierTester(subj_range=[1, 110], data_source='physionet', stim_select=combo[0], stim_type=combo[1],
                            result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], tmin=0, tmax=4, notch=60,
                            f_name=fname, f_path=fpath, random_state=rand, ch_list=['Fp1', 'Fp2', 'O1', 'O2'],
                            p_select=None, filter_bounds=(2., 60.))
    test.run_increment_batch_test(batch_size=30, incr_value=10, max_batch_size=109, n_times=5, sk_test=False,
                                  nn_test=True, split_subject=True, test_split=0.2)
    del test

#Motor Cortex Config
for combo in combinations:
    print("\nIterating for Combination: {c1}-{c2}\n\n".format(c1=combo[0], c2=combo[1]))
    # Form our path/filename. Here, we're saving somewhere different to the default to make them easy to find.
    fname = 'batch_{stim}_{type}_motor_cortex_nn'.format(stim=combo[0], type=combo[1])
    fpath = 'E:/PycharmProjects/OpenBCIResearch/CLassifierTesterResults/Batch/NN'

    # Form our testing class and run it.
    test = ClassifierTester(subj_range=[1, 110], data_source='physionet', stim_select=combo[0], stim_type=combo[1],
                            result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], tmin=0, tmax=4, notch=60,
                            f_name=fname, f_path=fpath, random_state=rand, ch_list=['C3', 'Cz', 'C4'],
                            p_select=None, filter_bounds=(2., 60.))
    test.run_increment_batch_test(batch_size=30, incr_value=10, max_batch_size=109, n_times=5, sk_test=False,
                                  nn_test=True, split_subject=True, test_split=0.2)
    del test
