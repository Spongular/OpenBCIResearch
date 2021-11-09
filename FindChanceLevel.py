

import mne
import numpy as np
import data_loading
import gen_tools

exceptions = [38, 80, 88, 89, 92, 100, 104]
r = range(1, 110)
chances = []

for sub in r:
    if sub in exceptions:
        continue

    #Grab the raws
    labels = []
    for x in range(1, 5):
        raw = data_loading.get_single_mi(sub, x)
        raw = gen_tools.preprocess_bandpass(raw)
        data, labs, epochs = gen_tools.epoch_data(raw, 1, 2, pick_list=['C3'])
        del data, epochs
        labels.append(labs)

    #Find the chance levels
    for labs in labels:
        class_balance = np.mean(labs == labs[0])
        class_balance = max(class_balance, 1. - class_balance)
        chances.append(class_balance)

print("Average Chance Level = {c}, std = {std}".format(c=np.mean(chances), std=np.std(chances)))

