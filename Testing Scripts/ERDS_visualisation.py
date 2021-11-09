
from os import listdir, path, remove
import fnmatch
import LiveBCI
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mne
import data_loading
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap

dataset = 'recorded'
live_layout = 'm_cortex'
sub = 2
stim_select = 'lr'

if dataset == 'recorded':

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
elif dataset == 'physionet':
    # 1 = left/right hands real.
    # 2 = left/right hands imagery.
    # 3 = hands/feet real.
    # 4 = hands/feet imagery.
    if stim_select == 'hf':
        stim = 3
    else:
        stim = 1
    raw = data_loading.get_single_mi(sub, stim)
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage("standard_1020", match_case=False)
    raw.rename_channels(lambda s: s.strip("."))
    events, event_ids = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))

# epoch data ##################################################################
picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])
tmin, tmax = -4, 4  # define epochs around events (in s)
if stim_select == 'hf':
    event_ids = dict(hands=2, feet=3)  # map event IDs to tasks
elif stim_select == 'lr':
    event_ids = dict(left=2, right=3)  # map event IDs to tasks

epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5,
                    picks=picks, preload=True) #baseline=None,

# compute ERDS maps ###########################################################
freqs = np.arange(2, 36, 1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = [-1, 0]  # baseline interval (in s)
cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None, out_type='mask')  # for cluster test

# Run TF decomposition overall epochs
tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                     use_fft=True, return_itc=False, average=False,
                     decim=2)
tfr.crop(tmin, tmax)
tfr.apply_baseline(baseline, mode="percent")
for event in event_ids:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [10, 10, 10, 1]})
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=-1,
                                     **kwargs)

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
                              axes=ax, colorbar=False, show=False, mask=mask,
                              mask_style="mask")

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if ch != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1])
    fig.suptitle("ERDS ({})".format(event))
    fig.show()
    plt.show()

df = tfr.to_data_frame(time_format=None, long_format=True)

# Map to frequency bands:
freq_bounds = {'_': 0,
               'delta': 3,
               'theta': 7,
               'alpha': 13,
               'beta': 35,
               'gamma': 140}
df['band'] = pd.cut(df['freq'], list(freq_bounds.values()),
                    labels=list(freq_bounds)[1:])

# Filter to retain only relevant frequency bands:
freq_bands_of_interest = ['delta', 'theta', 'alpha', 'beta']
df = df[df.band.isin(freq_bands_of_interest)]
df['band'] = df['band'].cat.remove_unused_categories()

# Order channels for plotting:
df['channel'].cat.reorder_categories(['C3', 'Cz', 'C4'], ordered=True,
                                     inplace=True)

g = sns.FacetGrid(df, row='band', col='channel', margin_titles=True)
g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=10)
axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
g.map(plt.axhline, y=0, **axline_kw)
g.map(plt.axvline, x=0, **axline_kw)
g.set(ylim=(-0.25, 3))
g.set_axis_labels("Time (s)", "ERDS (%)")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.add_legend(ncol=2, loc='lower center')
g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
g.fig.show()
plt.show()