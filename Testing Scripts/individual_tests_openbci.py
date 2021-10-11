from ClassifierTester import ClassifierTester
import random

#The sets of stimuli/operations to test.
combinations = [('hf', 'movement', 'HF-MM'), ('lr', 'movement', 'LR-MM')]

#Random state must be equal to compare effectively.
rand = random.randint(1, 999999)
r = range(1, 6)

for combo in combinations:
    print("\nIterating for Combination: {combo}\n\n".format(combo=combo[2]))
    for x in r:
        #Form our path/filename. Here, we're saving somewhere different to the default to make them easy to find.
        fname = 'sub{sub}_{stim}_{type}'.format(sub=x, stim=combo[0], type=combo[1])
        fpath = 'E:/PycharmProjects/OpenBCIResearch/CLassifierTesterResults/Individual/OpenBCI/{folder}'.format(folder=combo[2])

        #Form our testing class and run it.
        test = ClassifierTester(subj_range=[x, x+1], data_source='live-movement', stim_select=combo[0], stim_type=combo[1],
                                p_select='genetic', p_select_frac=1, result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'],
                                p_n_jobs=-1, tmin=-1, tmax=4, f_name=fname, f_path=fpath, random_state=rand,
                                live_layout='m_cortex')
        test.run_individual_test(sk_test=True, nn_test=False, cross_val_times=5)
        del test

        #Perform the same for the filter bank method
        fname = fname + '_fb'
        test = ClassifierTester(subj_range=[x, x + 1], data_source='live-movement', stim_select=combo[0],
                                stim_type=combo[1], filter_bank=True, p_select='genetic', p_select_frac=1,
                                result_metrics=['acc', 'f1', 'rec', 'prec', 'roc'], p_n_jobs=-1, tmin=-1, tmax=4,
                                f_name=fname, f_path=fpath, random_state=rand, live_layout='m_cortex')
        test.run_individual_test(sk_test=True, nn_test=False, cross_val_times=5)
        del test

        print('Test on Subject {sub} Completed'.format(sub=x))
