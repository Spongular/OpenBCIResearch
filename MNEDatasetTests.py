# https://mne.tools/0.11/manual/datasets_index.html#eegbci-motor-imagery

#First, import the entire mne thing.
import mne

#Set the path for our file. In this case, it's the first trial.
file = "EEGRecordings\\PhysioNetMMDB\\eegmmidb-1.0.0.physionet.org\\S001\\S001R03.edf"

#Use the read_raw_edf function to grab our little file
data = mne.io.read_raw_edf(file)

#Get the information.
raw_data = data.get_data()
info = data.info
channels = data.ch_names
annotations = data.annotations

#Print it to check.
print(info)
print(channels)
print(annotations)

# start, stop = raw.time_as_index([100, 115])
# data, times = raw[:, start:stop]
# print(data.shape)
# print(times.shape)
# data, times = raw[2:20:3, start:stop]