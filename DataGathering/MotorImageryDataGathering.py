import LiveBCI

import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from psychopy import visual, core

import mne

board = LiveBCI.generateOpenBCIBoard('COM3', 'ganglion', timeout=15)

#This data uses the Fp1, Fp2, O1, O2 electrode placements.
stimulator = LiveBCI.MotorImageryStimulator(stim_time=4, wait_time=6, stim_count=12, stim_type='lr', board=board,
                                            ch_count=4, ch_names=['Fp1', 'Fp2', 'O1', 'O2'])

raw = stimulator.run_stim(return_raw=True)

raw.plot_psd()
raw.plot(block=True)

stimulator.save_data(file_name='subject1-17_08_2021-mi-lr-004', type='imagery')