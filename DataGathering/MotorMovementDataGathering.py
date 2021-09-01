import LiveBCI

import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from psychopy import visual, core

import mne

board = LiveBCI.generateOpenBCIBoard('COM3', 'ganglion', timeout=15)

#This data uses the Fp1, Fp2, O1, O2 electrode placements.
stimulator = LiveBCI.MotorImageryStimulator(stim_time=4, wait_time=6, stim_count=12, stim_type='lr', board=board,
                                            ch_count=3, ch_names=['C3', 'Cz', 'C4'])
raw = stimulator.run_stim(return_raw=True)

raw.plot(block=True)

stimulator.save_data(file_name='subject1-27_08_2021-m_cortex_electrode_placement-mm-lr-004', type='movement')