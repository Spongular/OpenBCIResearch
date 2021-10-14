from ClassifierTester import ClassifierTester
import random

#The sets of stimuli/operations to test.
# combinations = [('hf', 'imaginary'), ('hf', 'movement'),
#                 ('lr', 'imaginary'), ('lr', 'movement')]
combinations = [('hf', 'imaginary')]
rand = random.randint(1, 999999)

#For Machine Learning---------------------------------------------------------------------------------------------------
#Full 64 Channels
for combo in combinations:
    print("\nIterating for Combination: {c1}-{c2}\n\n".format(c1=combo[0], c2=combo[1]))
    # Form our path/filename. Here, we're saving somewhere different to the default to make them easy to find.
    fname = 'batch_{stim}_{type}_64ch_ml'.format(stim=combo[0], type=combo[1])
    fpath = 'E:/PycharmProjects/OpenBCIResearch/CLassifierTesterResults/Batch/ML'

    # Form our testing class and run it.
    # test = ClassifierTester(subj_range=[1, 110], data_source='physionet', stim_select=combo[0], stim_type=combo[1],
    #                         result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], tmin=-1, tmax=4,
    #                         f_name=fname, f_path=fpath, random_state=rand, p_select='genetic', p_select_frac=0.15, p_n_jobs=-1)
    # test.run_increment_batch_test(batch_size=10, incr_value=10, max_batch_size=109, n_times=5, sk_test=True,
    #                               nn_test=False, split_subject=True)
    # del test

    #Do the same for filter bank.
    fname = fname + '_fb'
    test = ClassifierTester(subj_range=[1, 110], data_source='physionet', stim_select=combo[0], stim_type=combo[1],
                            result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], tmin=-1, tmax=4,
                            f_name=fname, f_path=fpath, random_state=rand, p_select=None, filter_bank=True)
    test.run_increment_batch_test(batch_size=10, incr_value=10, max_batch_size=109, n_times=5, sk_test=True,
                                  nn_test=False, split_subject=True)
    del test
quit()
#Headband Config
for combo in combinations:
    print("\nIterating for Combination: {c1}-{c2}\n\n".format(c1=combo[0], c2=combo[1]))
    # Form our path/filename. Here, we're saving somewhere different to the default to make them easy to find.
    fname = 'batch_{stim}_{type}_headband_ml'.format(stim=combo[0], type=combo[1])
    fpath = 'E:/PycharmProjects/OpenBCIResearch/CLassifierTesterResults/Batch/ML'

    # Form our testing class and run it.
    # test = ClassifierTester(subj_range=[1, 110], data_source='physionet', stim_select=combo[0], stim_type=combo[1],
    #                         result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], p_n_jobs=-1, tmin=-1, tmax=4,
    #                         f_name=fname, f_path=fpath, random_state=rand, ch_list=['Fp1', 'Fp2', 'O1', 'O2'],
    #                         p_select='genetic', p_select_frac=0.15)
    # test.run_increment_batch_test(batch_size=10, incr_value=10, max_batch_size=109, n_times=5, sk_test=True,
    #                               nn_test=False, split_subject=True)
    # del test

    #Do the same for filter bank.
    fname = fname + '_fb'
    test = ClassifierTester(subj_range=[1, 110], data_source='physionet', stim_select=combo[0], stim_type=combo[1],
                            result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], tmin=-1, tmax=4,
                            f_name=fname, f_path=fpath, random_state=rand, p_select=None,
                            filter_bank=True, ch_list=['Fp1', 'Fp2', 'O1', 'O2'])
    test.run_increment_batch_test(batch_size=10, incr_value=10, max_batch_size=109, n_times=5, sk_test=True,
                                  nn_test=False, split_subject=True)
    del test

#Motor Cortex Config
for combo in combinations:
    print("\nIterating for Combination: {c1}-{c2}\n\n".format(c1=combo[0], c2=combo[1]))
    # Form our path/filename. Here, we're saving somewhere different to the default to make them easy to find.
    fname = 'batch_{stim}_{type}_motor_cortex_ml'.format(stim=combo[0], type=combo[1])
    fpath = 'E:/PycharmProjects/OpenBCIResearch/CLassifierTesterResults/Batch/ML'

    # Form our testing class and run it.
    # test = ClassifierTester(subj_range=[1, 110], data_source='physionet', stim_select=combo[0], stim_type=combo[1],
    #                         result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], p_n_jobs=-1, tmin=-1, tmax=4,
    #                         f_name=fname, f_path=fpath, random_state=rand, ch_list=['C3', 'Cz', 'C4'],
    #                         p_select='genetic', p_select_frac=0.15)
    # test.run_increment_batch_test(batch_size=10, incr_value=10, max_batch_size=109, n_times=5, sk_test=True,
    #                               nn_test=False, split_subject=True)
    # del test

    # Do the same for filter bank.
    fname = fname + '_fb'
    test = ClassifierTester(subj_range=[1, 110], data_source='physionet', stim_select=combo[0], stim_type=combo[1],
                            result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], tmin=-1, tmax=4,
                            f_name=fname, f_path=fpath, random_state=rand, p_select=None,
                            filter_bank=True, ch_list=['C3', 'Cz', 'C4'])
    test.run_increment_batch_test(batch_size=10, incr_value=10, max_batch_size=109, n_times=5, sk_test=True,
                                  nn_test=False, split_subject=True)
    del test
