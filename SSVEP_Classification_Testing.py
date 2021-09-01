
import data_loading
import gen_tools
import numpy as np
from mne.time_frequency import tfr_morlet
from matplotlib import pyplot as plt

#Grab the data, bandpass and then check the psd.
raw = data_loading.get_ssvep_MAMEM('D:\\EEG Data\\mne')
raw.filter(1., 40., fir_design='firwin', skip_by_annotation='edge')

#Epoch the data
epochs = gen_tools.epoch_ssvep_MAMEM(raw=raw, tmin=-1., tmax=5., pick_list=['O1', 'O2', 'Oz'])

#Form Time Frequency Representations of the data.
freqs = np.linspace(start=5, stop=40, num=60, endpoint=True)
n_cycles = []
#We need to create a list of n_cycles so that we can deal with the differences
#in wavelet lengths. i.e. 1hz wavelet is 1 second, but a 20hz is 1/20th of a second.
for x in freqs:
    if x / 1.5 < 2:
        n = 2
    else:
        n = x / 2
    n = round(n)
    n_cycles.append(n)
#Now, for each type of event, we perform the tfr_morlet function and plot it.
for x in epochs.event_id:
    tfr, itc = tfr_morlet(epochs[x], freqs=freqs, picks='all', n_cycles=n_cycles, return_itc=True)
    tfr.plot(picks='all', baseline=(-0.5, -0.1), mode='logratio',
                 title='O1, O2, Oz - {hz} Hz stim'.format(hz=x));
    plt.show(block=True)

