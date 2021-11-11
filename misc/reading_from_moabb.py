#This file is for exploring, understanding and converting .mat files into MNE raw objects.
#This primarily exists because free datasets are invariably some form of .mat file because
#scientists are afraid of ever using a standardised format, and just save the entire set as
#their own 'perfectly reasonable' form within a .mat file. Can you tell that I'm annoyed?

#imports
from moabb import datasets
from mne import set_config, get_config, concatenate_raws, Epochs, find_events, pick_types
import matplotlib.pyplot as plt

#Make sure to set the data path to where we want our data to end up.
set_config("MNE_DATASETS_MAMEM1_PATH", 'D:\\EEG Data\\mne')
print("New data path: {config}".format(config=get_config("MNE_DATASETS_MAMEM1_PATH")))

print("\nLoading dataset...")
#Get the data from the moabb classes. This will automatically download if no data is present.
#Does not seem to be compatible with the public release of the MAMEN datasets by the original
#creators, but instead it's a modified set, likely to keep a consisten file format.
dataset = datasets.ssvep_mamem.MAMEM1()
data = dataset.get_data(dataset.subject_list)
print(data)

print("\nConverting to single RAW...")

#The 'data' attribute is in the form of a three-tier dictionary, i.e. data['1']['session_b']['run_0']
#and thus we need to iterate through it to concatenate all our raw data.
raws = []
for key_a in data:
    for key_b in data[key_a]:
        for key_c in data[key_a][key_b]:
            raws.append(data[key_a][key_b][key_c])

#Now delete data to save memory
del data

#Concatenate and delete our old raws.
raw = concatenate_raws(raws)
del raws
print(raw.info)
raw.plot(block=True)
#From what we can see here, the data itself is in a different format to what is expected.
#in all likelihood, the data is in uV when mne expects V. Thus, we need to scale it. Not sure how yet.

print("\nConverting to 10-20 system and dropping extraneous channels...")

#For our system, we want the 10-20 equivalents for the channels, not all 256, so we create a dict from the
#approximates given in the MAMEN documentation and then rename channels to match. Then, we make the right picks
#when it comes time to epoch. The channels are according to https://www.egi.com/images/HydroCelGSN_10-10.pdf
ch_dict = {'E59': 'C3', 'E34': 'AF3', 'E183': 'C4', 'E44': 'C1', 'E161': 'PO8', 'E36': 'F3', 'E12': 'AF4',
           'E224': 'F4', 'E20': 'Afz', 'E47': 'F7', 'E179': 'TP8', 'E2': 'F8', 'E42': 'FC3', 'E37': 'FP1',
           'E66': 'CP3', 'E18': 'FP2', 'E162': 'P6', 'E26': 'FPZ', 'E109': 'PO3', 'E21': 'Fz', 'E185': 'C2',
           'E116': 'O1', 'E24': 'FC1', 'E150': 'O2', 'E140': 'PO4', 'E87': 'P3', 'E126': 'Oz', 'E153': 'P4',
           'E143': 'Cp2', 'E207': 'FC2', 'E79': 'CP1', 'E94': 'TP9', 'E29': 'F1', 'E101': 'Pz', 'E15': 'Fcz',
           'E119': 'Poz', 'E190': 'TP10', 'E5': 'F2', 'E226': 'F10', 'E49': 'FC5', 'E142': 'P2', 'E219': 'Ft10',
           'E48': 'F5', 'E194': 'C6', 'E106': 'P9', 'E67': 'Ft9', 'E206': 'FC4', 'E222': 'F6', 'E76': 'Cp5',
           'E211': 'FT8', 'E213': 'FC6', 'E10': 'AF8', 'E81': 'CpZ', 'E97': 'Po7', 'E172': 'CP6',
           'E46': 'AF7', 'E64': 'C5', 'E164': 'CP4', 'E84': 'Tp7', 'E169': 'P10', 'E62': 'FT7',
           'E252': 'F9', 'E68': 'T9', 'E88': 'P1', 'E210': 'T10', 'E86': 'P5', 'E69': 'T7', 'E96': 'P7',
           'E202': 'T8', 'E170': 'P8'}
raw.rename_channels(ch_dict)
selections = ['C3', 'AF3', 'C4', 'C1', 'PO8', 'F3', 'AF4',
              'F4', 'Afz', 'F7', 'TP8', 'F8', 'FC3', 'FP1',
              'CP3', 'FP2', 'P6', 'FPZ', 'PO3', 'Fz', 'C2',
              'O1', 'FC1', 'O2', 'PO4', 'P3', 'Oz', 'P4',
              'Cp2', 'FC2', 'CP1', 'TP9' 'F1', 'Pz', 'Fcz',
              'Poz', 'TP10' 'F2', 'F10', 'FC5', 'P2', 'Ft10',
              'F5', 'C6', 'P9', 'Ft9', 'FC4', 'F6', 'Cp5',
              'FT8' 'FC6', 'AF8', 'CpZ', 'Po7', 'CP6',
              'AF7', 'C5', 'CP4', 'Tp7', 'P10', 'FT7',
              'F9', 'T9', 'P1', 'T10', 'P5', 'T7', 'P7',
              'T8', 'P8', 'stim']
raw.pick_channels(selections)
raw.plot(block=True)

#From what we can see here, the data itself is in a different format to what is expected.
#in all likelihood, the data is in uV when mne expects V. Thus, we need to scale it.

print("\nConverting from uV to V...")
raw.apply_function(fun=lambda x: x / 1e6, picks=['eeg'], n_jobs=2)
raw.plot(block=True)

print("\nPerforming bandpass filter in range {min} to {max}".format(min=4, max=35))
raw.filter(4., 35., fir_design='firwin', skip_by_annotation='edge')
raw.plot_psd(fmin=4., fmax=35.)
raw.plot(block=True)

print("\nEpoching...")
#In the stim channel, our events are: "6.66": 1, "7.50": 2, "8.57": 3, "10.00": 4, "12.00": 5
#And each stim moment lasts for five seconds. Thus, we find the epochs as...
event_id = {"6.66": 1, "7.50": 2, "8.57": 3, "10.00": 4, "12.00": 5}
events = find_events(raw=raw, stim_channel='stim')
tmin = 0.
tmax = 5.
#Additionally, for our system, we want the 10-20 equivalents for the channels, not all 256, so...
ssvep_epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=None, preload=True)
ssvep_epochs.plot_psd
ssvep_epochs.plot(block=True)

print("\nCreating evoked data...")

evoked_6 = ssvep_epochs['6.66'].average()
evoked_7 = ssvep_epochs['7.50'].average()
evoked_8 = ssvep_epochs['8.57'].average()
evoked_10 = ssvep_epochs['10.00'].average()
evoked_12 = ssvep_epochs['12.00'].average()

evoked_6.plot()
evoked_7.plot()
evoked_8.plot()
evoked_10.plot()
evoked_12.plot()
plt.show(block=True)
