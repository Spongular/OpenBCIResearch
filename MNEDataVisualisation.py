
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
import matplotlib.pyplot as plt

#Set our log level to avoid excessive bullshit.
import gen_tools

mne.set_log_level("WARNING")

#For a single file:
#file = "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S001\\S001R04.edf"
#raw = mne.io.read_raw_edf(file, preload=True)

#For multiple:
#raw = data_loading.get_single_mi(1, 2)

#For All:
raw = data_loading.get_all_mi_between(1, 40, 2, ["088", "092", "100"])

#This file is for the imagined opening and closing of the left and right fists.

#Read the raw. We need to remove the '.' from channel names and set a montage for visualising.
#raw = mne.io.read_raw_edf(file, preload=True)
raw.rename_channels(lambda s: s.strip("."))
raw.set_montage("standard_1020", match_case=False)
raw.set_eeg_reference("average", projection=True)

#Now, we can simply plot the data.
#order = np.arange(raw.info['nchan'])
#raw.plot(n_channels=10, order=order, block=True)

#Let's take a look at the power spectral density
#raw.plot_psd()

#It's not all that useful in that form though, so let's epoch it.
events, event_dict = mne.events_from_annotations(raw)

#Check our events.
print(events)
print(event_dict)

#Here we can plot our events to see when they occur.
#event_plot = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)

#From the dataset details, we know that T1 is the imagined onset of motion for
#the left fist, and that T2 is for the right. So, we remove T0 as it's not relevant.
del event_dict['T0']

#Now, we can create and plot some Epochs
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-1, tmax=4,
                    preload=True)

print(epochs.event_id)
#epochs['T2'].plot()
#epochs['T1'].plot(block=True)

#We can now average them to see the evoked response.
evoked_t1 = epochs['T1'].average()
evoked_t2 = epochs['T2'].average()

# evoked_t1.plot()
# evoked_t2.plot()

#However, we can see that the data is a little bit difficult to understand.
#When processing the data for classification, we make use of bandpass filters
#to ensure that the data stays within the expected frequencies. So, let's do that.
#In this case, we stick to the 'Mu' band of 8-12hz.
raw_f = raw.filter(6., 30., fir_design='firwin')
epochs = mne.Epochs(raw_f, events, event_id=event_dict, tmin=-0.2, tmax=0.5,
                    preload=True)
evoked_t1 = epochs['T1'].average()
evoked_t2 = epochs['T2'].average()

epochs.plot_psd(fmin=6., fmax=30., )
epochs.plot_psd_topomap(normalize=True)

#epochs.plot(block=True)

freqs = np.logspace(*np.log10([6, 30]), num=10)
n_cycles = freqs / 3.  # different number of cycle per frequency
power_t1= tfr_morlet(evoked_t1, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, decim=3, n_jobs=1)
power_t2= tfr_morlet(evoked_t2, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, decim=3, n_jobs=1)
power_epochs = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, decim=3,
                          n_jobs=1, average=False, return_itc=False)

power_t1.plot_joint(baseline=(-0.5, 0), mode='mean', tmin=-0.2, tmax=0.5)
power_t2.plot_joint(baseline=(-0.5, 0), mode='mean', tmin=-0.2, tmax=0.5)
power_epochs.plot()

print(power_t1.data.shape)
print(power_epochs.data.shape)

#evoked_t1.plot_topomap()
evoked_t1.plot(spatial_colors=True, gfp=True)
#evoked_t2.plot_topomap()
evoked_t2.plot(spatial_colors=True, gfp=True)

#X, Y = gen_tools.wavelet_transform(epochs, f_low=6., f_high=30., f_num=10)


# evoked_t1.plot(picks=['C3', 'Cz', 'C4'], spatial_colors=True)
# evoked_t2.plot(picks=['C3', 'Cz', 'C4'], spatial_colors=True)

#From the plots, we can see a clear drop in oscillation for T2, and an increase for T1, that
#both begin around 0.1-0.2 seconds after the marker is shown to the person.

#Let's try applying Current Source Density to the issue.
# raw_csd = mne.preprocessing.compute_current_source_density(raw)
# raw_csd.filter(8., 12., fir_design='firwin')
# epochs = mne.Epochs(raw_f, events, event_id=event_dict, tmin=-0.2, tmax=0.5,
#                     preload=True)
# evoked_t1_csd = epochs['T1'].average()
# evoked_t2_csd = epochs['T2'].average()
#
# #Get rid of unnecessary data
# del raw_csd
#
# epochs.plot(block=True)

# evoked_t1_csd.plot(picks=['C3', 'Cz', 'C4'], spatial_colors=True, gfp=True)
# evoked_t2_csd.plot(picks=['C3', 'Cz', 'C4'], spatial_colors=True, gfp=True)



#We can also check the grand average if we want. Not really needed, but eh.
#grand_average = mne.grand_average([evoked_t1_csd, evoked_t2_csd])
#grand_average.plot(picks=['C3', 'Cz', 'C4'], spatial_colors=True)

#So, from the plots of C3, Cz and C4, we can see some clear change in behaviour
#At the 0.5 second point and onwards, with T1 causing a spike in activity, and T2
#causing a slight lull. This may, or may not, be enough to properly classify.
#Either way, we need to reduce the feature space if we want a good result.
#That's where Principal Component Analysis comes in.


