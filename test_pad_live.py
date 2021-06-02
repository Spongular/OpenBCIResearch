#This is used for testing the connection/reading of OpenBCI hardware.
import time
import matplotlib
import numpy as np
import psychopy
import LiveBCI

import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from psychopy import visual, core

import mne

board = LiveBCI.generateOpenBCIBoard('COM3', 'ganglion', timeout=15)

test = LiveBCI.MotorImageryStimulator(stim_time=4, wait_time=4, stim_count=5, stim_type='lr', board=board)

# raw = test.run_stim(return_raw=True)
#
# raw.plot(block=True)
#
# print(raw)
#
# test.save_data(file_name='test_ganglion_01')

test.load_data(file='LiveRecordings\\MotorImagery\\test_ganglion_01_raw.fif')


test.eeg_to_epochs(tmin=0., tmax=4., event_dict=dict(T1=1, T2=2), stim_ch='STIM001').plot(block=True)

test.epochs.plot_psd()

test.clear_epochs()

test.filter_raw(min=4.)

test.eeg_to_epochs(tmin=0., tmax=4., event_dict=dict(T1=1, T2=2), stim_ch='STIM001').plot(block=True)

test.epochs.plot_psd(fmin=4.)

test.epochs_to_evoked().plot()