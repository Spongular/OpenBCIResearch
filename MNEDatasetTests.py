# https://mne.tools/0.11/manual/datasets_index.html#eegbci-motor-imagery

#First, import the entire mne thing.
import mne

#Set the path for our file. In this case, it's the first trial.
file = "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S001\\S001R04.edf"

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
data.plot(block=True)

data.plot(n_channels=64, duration=5, scalings={"eeg": 75e-6}, start=10, block=True)


# start, stop = raw.time_as_index([100, 115])
# data, times = raw[:, start:stop]
# print(data.shape)
# print(times.shape)
# data, times = raw[2:20:3, start:stop]