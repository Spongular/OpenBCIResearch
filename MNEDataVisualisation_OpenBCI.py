
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


#options
live_layout = 'm_cortex'
sub = 2
stim_select = 'lr'

#Grab the file names and filter to match what is needed.
rootpath = 'E:\\PycharmProjects\\OpenBCIResearch\\DataGathering\\LiveRecordings\\MotorResponses\\Movement\\'
file_paths = listdir(rootpath)
if live_layout == 'headband':
    files = fnmatch.filter(file_paths, 'subject{sub}-??_??_????-mm-{stim}*'.format(sub=sub,
                                                                                     stim=stim_select))
elif live_layout == 'm_cortex':
    files = fnmatch.filter(file_paths, 'subject{sub}-??_??_????-m_cortex_electrode_placement-mm-{stim}*'.format(sub=sub,
                                                                                     stim=stim_select))
else:
    raise Exception("Error: 'live_layout' must be either 'headband' or 'm_cortex'")

#Format the file paths and load them through LiveBCI
file_paths = []
for file_name in files:
    file_paths.append(rootpath + file_name)
dloader = LiveBCI.MotorImageryStimulator(stim_time=4, wait_time=4, stim_count=5, stim_type='lr',
                                         board=None)
dloader.load_multiple_data(files=file_paths)
raw = dloader.raw
print(raw.info)
picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])
events = mne.find_events(raw=raw, stim_channel='STI001')

raw.plot(block=True)
raw.plot_psd(average=True, fmax=70.)

raw = gen_tools.preprocess_bandpass(raw, min=2., max=35.)

raw.plot(block=True)

dloader.raw = raw
dloader.eeg_to_epochs(tmin=-1, tmax=4, event_dict=dict(T1=2, T2=3), stim_ch='STI001')
epochs = dloader.epochs

epochs.plot(block=True)
epochs.plot_psd(average=True, fmin=2., fmax=35.)

print("T1 = {t1}, \nT2 = {t2}".format(t1=epochs['T1'], t2=epochs['T2']))

data = epochs.get_data()
labels = epochs.events[:, -1] - 2
print(data.shape)
print(labels.shape)
print(labels)