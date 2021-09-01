#Sources:
# https://mne.tools/0.11/manual/datasets_index.html#eegbci-motor-imagery
# https://mne.tools/dev/auto_tutorials/intro/plot_10_overview.html#sphx-glr-auto-tutorials-intro-plot-10-overview-py
# http://autoreject.github.io/


#First, import the entire mne thing.
import mne
#Autoreject is useful for cleaning epochs.
from autoreject import AutoReject

#Set the path for our file. In this case, it's the first trial.
file = "../EEGRecordings/PhysioNetMMDB/eegmmidb-1.0.0.physionet.org/S001/S001R04.edf"

mne.set_log_level("WARNING")

#Use the read_raw_edf function to grab our little file
data = mne.io.read_raw_edf(file, preload=True)
data.rename_channels(lambda s: s.strip("."))
data.set_montage("standard_1020", match_case=False)
data.set_eeg_reference("average")

#Get the information.
info = data.info
channels = data.ch_names
annotations = data.annotations



#Print it to check.
print(info)
print(channels)
print(annotations)

#These are a few ways to plot the data.
#Always make sure that block=True in scripts, as it will instantly close otherwise.
#data.plot(block=True)
#data.plot(n_channels=64, duration=5, scalings={"eeg": 75e-6}, start=10, block=True)

#Let's grab our events.
#In this instance, we use 'events_from_annotations' as there is no dedicated event channel.
events, event_dict = mne.events_from_annotations(data)

#Check our events.
print(events)
print(event_dict)

#Here we can plot our events to see when they occur.
event_plot = mne.viz.plot_events(events, event_id = event_dict, sfreq=data.info['sfreq'],
                                 first_samp = data.first_samp)

#Now, we can create some Epochs
epochs = mne.Epochs(data, events, event_id = event_dict, tmin=-0.2, tmax=0.5,
                    preload=True)
print(epochs)

#And then clean them.
auto_reject = AutoReject()
epochs_clean = auto_reject.fit_transform(epochs)

print(epochs_clean)

#Create a list of the events to use.
events_to_plot = ['T1', 'T2']

#Equalise our counts to avoid bias.
epochs_clean.equalize_event_counts(events_to_plot)

#Get our left and right
c_epochs_l = epochs_clean['T1']
c_epochs_r = epochs_clean['T2']

#Free up memory.
del data, epochs, epochs_clean

c_epochs_l.plot_image()
c_epochs_r.plot_image()


