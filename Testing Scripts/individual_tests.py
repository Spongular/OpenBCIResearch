from ClassifierTester import ClassifierTester
import random

#The sets of stimuli/operations to test.
combinations = [('hf', 'imaginary', 'HF-IM'), ('hf', 'movement', 'HF-MM'),
                ('lr', 'imaginary', 'LR-IM')]#, ('lr', 'movement', 'LR-MM')]
#The subjects to ignore due to bad data.
exceptions = [38, 80, 88, 89, 92, 100, 104]

#Random state must be equal to compare effectively.
#rand = random.randint(1, 999999)
rand = 108400
r = range(1, 110)

#Full 64 Channels
for combo in combinations:
    print("\nIterating for Combination: {combo}\n\n".format(combo=combo[2]))
    for x in r:
        #Skip our bad subjects.
        if x in exceptions:
            continue

        #Form our path/filename. Here, we're saving somewhere different to the default to make them easy to find.
        fname = 'sub{sub}_{stim}_{type}_64ch'.format(sub=x, stim=combo[0], type=combo[1])
        fpath = 'E:/PycharmProjects/OpenBCIResearch/CLassifierTesterResults/Individual/{folder}'.format(folder=combo[2])

        #Form our testing class and run it.
        # test = ClassifierTester(subj_range=[x, x+1], data_source='physionet', stim_select=combo[0], stim_type=combo[1],
        #                         p_select='genetic', p_select_frac=1, result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'],
        #                         p_n_jobs=-1, tmin=-1, tmax=4, f_name=fname, f_path=fpath, random_state=rand)
        # test.run_individual_test(sk_test=True, nn_test=False, cross_val_times=5)
        # del test

        #Perform the same for the filter bank method
        fname = fname + '_fb'
        test = ClassifierTester(subj_range=[x, x + 1], data_source='physionet', stim_select=combo[0],
                                stim_type=combo[1], filter_bank=True, p_select='genetic', p_select_frac=1,
                                result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], p_n_jobs=-1, tmin=-1, tmax=4,
                                f_name=fname, f_path=fpath, random_state=rand)
        test.run_individual_test(sk_test=True, nn_test=False, cross_val_times=5)
        del test

        print('Test on Subject {sub} Completed'.format(sub=x))

#Headband Config
for combo in combinations:
    print("\nIterating for Combination: {combo}\n\n".format(combo=combo[2]))
    for x in r:
        #Skip our bad subjects.
        if x in exceptions:
            continue

        #Form our path/filename. Here, we're saving somewhere different to the default to make them easy to find.
        fname = 'sub{sub}_{stim}_{type}_headband'.format(sub=x, stim=combo[0], type=combo[1])
        fpath = 'E:/PycharmProjects/OpenBCIResearch/CLassifierTesterResults/Individual/{folder}'.format(folder=combo[2])

        #Form our testing class and run it.
        # test = ClassifierTester(subj_range=[x, x+1], data_source='physionet', stim_select=combo[0], stim_type=combo[1],
        #                         p_select='genetic', p_select_frac=1, result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'],
        #                         p_n_jobs=-1, tmin=-1, tmax=4, f_name=fname, f_path=fpath,
        #                         ch_list=['Fp1', 'Fp2', 'O1', 'O2'], random_state=rand)
        # test.run_individual_test(sk_test=True, nn_test=False, cross_val_times=5)
        # del test

        #Perform the same for the filter bank method
        fname = fname + '_fb'
        test = ClassifierTester(subj_range=[x, x + 1], data_source='physionet', stim_select=combo[0],
                                stim_type=combo[1], filter_bank=True, p_select='genetic', p_select_frac=1,
                                result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], p_n_jobs=-1, tmin=-1, tmax=4,
                                f_name=fname, f_path=fpath, ch_list=['Fp1', 'Fp2', 'O1', 'O2'], random_state=rand)
        test.run_individual_test(sk_test=True, nn_test=False, cross_val_times=5)
        del test

        print('Test on Subject {sub} Completed'.format(sub=x))

#Motor Cortex Config
for combo in combinations:
    print("\nIterating for Combination: {combo}\n\n".format(combo=combo[2]))
    for x in r:
        #Skip our bad subjects.
        if x in exceptions:
            continue

        #Form our path/filename. Here, we're saving somewhere different to the default to make them easy to find.
        fname = 'sub{sub}_{stim}_{type}_motor_cortex'.format(sub=x, stim=combo[0], type=combo[1])
        fpath = 'E:/PycharmProjects/OpenBCIResearch/CLassifierTesterResults/Individual/{folder}'.format(folder=combo[2])

        #Form our testing class and run it.
        # test = ClassifierTester(subj_range=[x, x+1], data_source='physionet', stim_select=combo[0], stim_type=combo[1],
        #                         p_select='genetic', p_select_frac=1, result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'],
        #                         p_n_jobs=-1, tmin=-1, tmax=4, f_name=fname, f_path=fpath, ch_list=['C3', 'Cz', 'C4'],
        #                         random_state=rand)
        # test.run_individual_test(sk_test=True, nn_test=False, cross_val_times=5)
        # del test

        #Perform the same for the filter bank method
        fname = fname + '_fb'
        test = ClassifierTester(subj_range=[x, x + 1], data_source='physionet', stim_select=combo[0],
                                stim_type=combo[1], filter_bank=True, p_select='genetic', p_select_frac=1,
                                result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], p_n_jobs=-1, tmin=-1, tmax=4,
                                f_name=fname, f_path=fpath, ch_list=['C3', 'Cz', 'C4'], random_state=rand)
        test.run_individual_test(sk_test=True, nn_test=False, cross_val_times=5)
        del test

        print('Test on Subject {sub} Completed'.format(sub=x))
