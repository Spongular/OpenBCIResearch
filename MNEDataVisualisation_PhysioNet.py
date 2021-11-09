
import mne
import scipy
from mne import concatenate_raws, events_from_annotations, pick_types, Epochs
from mne.channels import make_standard_montage
from mne.datasets.eegbci import eegbci
from mne.decoding import CSP
from mne.io import read_raw_edf

import numpy as np
from mne.time_frequency import psd_welch, psd_multitaper, tfr_morlet

import data_loading
import gen_tools
import matplotlib.pyplot as plt
from os import listdir, path, remove
import fnmatch
import LiveBCI
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mne
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap

#Set our log level to avoid excessive bullshit.
mne.set_log_level("WARNING")


# #For All:
# raw = data_loading.get_all_mi_between(1, 2, 2, ["088", "092", "100"])
# raw.plot(picks=['C3'], block=True)
# #This file is for the imagined opening and closing of the left and right fists.
#
# #Read the raw. We need to remove the '.' from channel names and set a montage for visualising.
# raw.rename_channels(lambda s: s.strip("."))
# raw.set_montage("standard_1020", match_case=False)
# raw.set_eeg_reference("average", projection=True)
#
# #Now, we can simply plot the data.
# order = np.arange(raw.info['nchan'])
# raw.plot(order=order, block=True)
#
# #Let's take a look at the power spectral density
# raw.plot_psd()

# This just sets our values for what kind of data we want.
# 1 = left/right hands real.
# 2 = left/right hands imagery.
# 3 = hands/feet real.
# 4 = hands/feet imagery.
def __get_runs(test_type):
    run = None
    if test_type == 1:
        run = 3
    elif test_type == 2:
        run = 4
    elif test_type == 3:
        run = 5
    elif test_type == 4:
        run = 6
    return run

mne.set_log_level('WARNING')
tmin, tmax = -1., 4. #This determines the boundaries for the epochs.
test = 1
#event_dict = dict(hands=2, feet=3) #Define our events.


subjects = set(range(30, 31))
exclusions = set([38, 80, 88, 89, 92, 100, 104])
subjects = subjects - exclusions

#Select the hands & feet tests.
files = []
run = __get_runs(test)
for a in subjects:
    num = "{0:03}".format(a)
    files += ["EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R" + "{:02d}".format(run) + ".edf",
              "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R" + "{:02d}".format(run + 4) + ".edf",
              "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S" + num + "\\S" + num + "R" + "{:02d}".format(run + 8) + ".edf"]

#load the files.
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])

#This essentially ensures that the channel names meet the expected standard.
eegbci.standardize(raw)

#You're gonna need a montage.
montage = make_standard_montage('standard_1020')
raw.set_montage(montage)

#Stripe the "." character from channels.
raw.rename_channels(lambda x: x.strip('.'))

raw.plot(block=True)
raw.plot_psd(average=True, fmax=70., picks=['C3', 'Cz', 'C4'])

#Apply a band-pass filter.
#In this instance, the lower bound is 7hz, and upper is 30hz.
#The skip_by_annotation setting catches any leftovers from the concat.
raw.filter(2., 35., fir_design='firwin', skip_by_annotation='edge')

#Now we grab events.
events, event_id = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

#Next, we pick our channels by type and name.
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

#Now we can finally get our epochs. Training will be done between 1s and 2s in the epochs.
epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks, baseline=None, preload=True)
epochs.plot_psd(average=True, fmin=2., fmax=35., picks=['C3', 'Cz', 'C4'])
epochs_train = epochs.copy()#.crop(tmin = 1., tmax = 2.)
labels = epochs.events[:,-1] -2

csp = CSP(n_components=4, reg='ledoit_wolf')
data = epochs.get_data()
csp.fit_transform(data, labels)

csp.plot_patterns(epochs.info)
plt.show(block=True)

print(data.shape)