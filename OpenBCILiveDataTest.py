#This script is primarily for testing the functions of the
#Brainflow library with MNE, and the PsychoPy library for
#presenting stimuli.
import time
import matplotlib
import numpy as np
import psychopy

import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from psychopy import visual, core

import mne

#Set log level.
BoardShim.enable_dev_board_logger()

#Generate stimulus window.
win = visual.Window([400, 400]) #View window of 400x400
msg = visual.TextStim(win, text='Beginning Synthetic\nTest') #Message to start test.
msg.autoDraw = True #Automatically draw msg every frame.
win.flip()
core.wait(2.0)


# use synthetic board for demo
params = BrainFlowInputParams()
board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
board.prepare_session()
board.start_stream()

print(board.get_sampling_rate(board.board_id))

#Add a 1 to the marker channel every second for ten seconds.
for i in range(10):
    msg.text = 'Awaiting Stimuli...'
    win.flip()
    time.sleep(1)
    msg.text = 'Stimulus Period'
    win.flip()
    board.insert_marker(1)
    time.sleep(1)
data = board.get_board_data()
core.wait(2)
data = np.concatenate((data, board.get_board_data()), axis=1)
board.stop_stream()
board.release_session()

msg.text = 'Session Over\nProcessing Data...'
win.flip()

eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
eeg_data = data[eeg_channels, :]
eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE
#Add the marker/stim channel to the data.
eeg_data = np.concatenate((eeg_data, data[[BoardShim.get_marker_channel(BoardIds.SYNTHETIC_BOARD.value)], :]), axis=0)

# Creating MNE objects from brainflow data arrays
ch_types = ['eeg'] * len(eeg_channels)
ch_names = BoardShim.get_eeg_names(BoardIds.SYNTHETIC_BOARD.value)
#Need to append 'stim' info.
ch_types.append('stim')
ch_names.append('STI001')
sfreq = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD.value)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(eeg_data, info)

#Plot the data.
raw.plot(block=True, show=True)
raw.copy().pick_types(eeg=False, stim=True).plot(block=True)

print(data)
