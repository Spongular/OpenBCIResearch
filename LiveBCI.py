#This script contains all the methods needed for live BCI recording, visualisation and decoding.

#System Imports
import time
from random import shuffle

#Brainflow Imports
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

#Assorted Math/Numerical Imports
import matplotlib
import numpy as np

#MNE EEG Decoding Imports
import mne

#Machine Learning Imports

#PsychoPi EEG Stimulation Imports
from psychopy import visual, core

def generateSyntheticBoard():
    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    return board

#Note: May need to apply driver fix if there is latency.
#https://docs.openbci.com/docs/10Troubleshooting/FTDI_Fix_Windows
#Serial Port information can be found in Device Manager. Should be in form COM*.
def generateOpenBCIBoard(serial_port, type='ganglion', mac_address=None, timeout=15):
    #Create params.
    params = BrainFlowInputParams()

    if type == 'ganglion':
        params.serial_port = serial_port #This is the port for the USB.
        params.timeout = timeout
        if mac_address is not None:
            params.mac_address = mac_address
        board = BoardShim(1, params)
        return board
    else:
        raise Exception("Error: Invalid board type selected. Only 'ganglion' is currently supported.")

#This is the stimulator for motor imagery.
class MotorImageryStimulator:
    def __init__(self, stim_time, wait_time, stim_count, stim_images=[], stim_image_size=[600,600], stim_type='lr',
                 board=None, ch_count=4, ch_names=[], classifier=None, classifier_type=None, output_location='DEFAULT',
                 visualize=False):
        #Stimulation Attributes
        self.stim_time = stim_time
        self.wait_time = wait_time
        self.stim_count = stim_count
        self.stim_names = []
        self.stim_images = []

        #If we are not passed a list of image locations, we use the default.
        if stim_images == []:
            self.stim_images = [] #This will hold the strings for the location of the stim images.
            self.stim_images.append('StimImages\\stim_empty.png') #This is our 'blank canvas'
            self.stim_names.append('WAIT')
            if stim_type == 'lr':
                self.stim_images.append('StimImages\\stim_left.png')
                self.stim_names.append('LEFT')
                self.stim_images.append('StimImages\\stim_right.png')
                self.stim_names.append('RIGHT')
            elif stim_type == 'hf':
                self.stim_images.append('StimImages\\stim_up.png')
                self.stim_names.append('HANDS')
                self.stim_images.append('StimImages\\stim_down.png')
                self.stim_names.append('FEET')
            else:
                raise Exception("Error: stim_type must be either 'lr' or 'hf' when stim_images = []")
        else:
            if len(stim_images) < 2:
                raise Exception("Error: stim_images must either be [], or have at least two elements: A waiting image" +
                                " and at least one stimulus image")
            self.stim_images = stim_images
            # Since we have custom images, we use T0, T1, T2... as names instead.
            for x in range(0, len(self.stim_images)):
                self.stim_names.append("T{num}".format(num=x))

        if stim_image_size[0] < 1 or stim_image_size[1] < 1:
            raise Exception("Error: stim_image_size must be an array of two elements, both being greater than 1.")
        self.stim_image_size = stim_image_size

        #EEG Input Attributes
        self.board = board
        self.ch_count = ch_count
        self.ch_names = ch_names

        #If our channel names aren't provided, we generate generic ones.
        if len(self.ch_names) < 1:
            print("No channel names provided. Using generic names")
            for x in range(0, ch_count):
                self.ch_names.append("CH{num}".format(num=x))

        #And append our stim channel name.
        self.ch_names.append("STIM001")

        #Classifier Attributes
        self.classifier = classifier
        self.classifier_type = classifier_type

        #Check the type to ensure it's correct.
        if self.classifier_type != 'sk-learn' and self.classifier_type != 'keras'\
                and self.classifier_type is not None:
            raise Exception("Error: classifier_type must be either 'sk-learn', 'keras' or None")

        #IO and Visualising Attributes
        self.output_location = output_location
        self.visualize = visualize
        self.eeg_data = None

    #Attribute Changing Functions
    def set_board(self, board):
        self.board = board
        return

    def set_eeg_details(self, ch_count, ch_names):
        if ch_count != len(ch_names):
            raise Exception("Error: ch_count must be equal to the length of ch_names")
        self.ch_count = ch_count
        self.ch_names = ch_names

    def set_classifier_details(self, classifier, classifier_type):
        if classifier is None and classifier_type is None:
            self.classifier = None
            self.classifier_type = None
        elif classifier is not None and classifier_type is not None:
            if classifier_type != 'sk-learn' and classifier_type != 'keras':
                raise Exception("Error: classifier_type must be either 'sk-learn' or 'keras'")
            self.classifier = classifier
            self.classifier_type = classifier_type
        else:
            raise Exception("Error: classifier and classifier_type must either both be None, or both be valid values")
        return

    def set_output_location(self, output_location):
        self.output_location = output_location

    def eeg_to_raw(self):
        #Check to make sure we have data.
        if self.eeg_data is None:
            raise Exception("Error: No EEG data has been recorded.")

        #Form an MNE Raw object from our data.
        eeg_channels = self.board.get_eeg_channels(self.board.board_id)
        eeg_data = self.eeg_data[eeg_channels, :]
        eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE
        # Add the marker/stim channel to the data.
        eeg_data = np.concatenate((eeg_data, eeg_data[[self.board.get_marker_channel(self.board.board_id)], :]),
                                  axis=0)

        # Creating MNE objects from brainflow data arrays
        ch_types = ['eeg'] * len(eeg_channels)
        # Need to append 'stim' info.
        ch_types.append('stim')
        sfreq = self.board.get_sampling_rate(self.board.board_id)
        info = mne.create_info(ch_names=self.ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(eeg_data, info)

        return raw


    #This is used to test the stimulation window.
    #Does not read or record from board.
    def test_stim(self):
        #Start up the stimulation window and display initial text.
        stim_win = visual.Window(self.stim_image_size)
        msg = visual.TextStim(stim_win, text='Initiating Testing Run...')  # Message to start test.
        msg.autoDraw = True  # Automatically draw msg every frame.
        stim_win.flip()

        #Set values for stims.
        stim_bag = []
        for x in range(0, self.stim_count):
            for y in range(1, len(self.stim_images)):
                stim_bag.append(y)
        shuffle(stim_bag)

        #Turn off autodraw for the message.
        core.wait(2.0)
        msg.setAutoDraw(False)
        stim_win.flip()

        # Display the non-stim image
        im = visual.ImageStim(stim_win, self.stim_images[0])
        im.setAutoDraw(True)
        stim_win.flip()

        #While our stim_bag has elements, we run through them.
        while(len(stim_bag) > 0):
            #Ensure the non-stim image is displayed and then wait.
            im.image = self.stim_images[0]
            stim_win.flip()
            core.wait(self.wait_time)

            #Grab our stim from the bag and then wait for the stim time.
            im.image = self.stim_images[stim_bag.pop()]
            stim_win.flip()
            core.wait(self.stim_time)

        #Display the non-stim for a final wait time.
        im.image = self.stim_images[0]
        stim_win.flip()
        core.wait(self.wait_time)

        #Remove image and display final message
        im.setAutoDraw(False)
        stim_win.flip()
        msg.text = "Testing Run Complete..."
        msg.setAutoDraw(True)
        stim_win.flip()
        core.wait(3)
        stim_win.close()
        return

    #This is the primary function for recording
    #eeg data. Presents the stimulation and records the data
    #to a mne raw object to return
    def run_stim(self, return_raw=False):
        # Start up the stimulation window and display initial text.
        stim_win = visual.Window(self.stim_image_size)
        msg = visual.TextStim(stim_win, text='Initiating Testing Run...')  # Message to start test.
        msg.autoDraw = True  # Automatically draw msg every frame.
        stim_win.flip()

        # Set values for stims.
        stim_bag = []
        for x in range(0, self.stim_count):
            for y in range(1, len(self.stim_images)):
                stim_bag.append(y)
        shuffle(stim_bag)

        # Turn off autodraw for the message.
        core.wait(2.0)
        msg.setAutoDraw(False)
        stim_win.flip()

        # Display the non-stim image
        im = visual.ImageStim(stim_win, self.stim_images[0])
        im.setAutoDraw(True)
        stim_win.flip()

        #Prepare the data stream
        self.board.prepare_session()
        self.board.start_stream()
        core.wait(2.0)
        #Empty the brainflow ring buffer into eeg_data
        self.eeg_data = self.board.get_board_data()

        # While our stim_bag has elements, we run through them.
        while (len(stim_bag) > 0):
            # Ensure the non-stim image is displayed and then wait.
            im.image = self.stim_images[0]
            stim_win.flip()
            core.wait(self.wait_time)

            #Set stim image, marker and wait.
            stim = stim_bag.pop()
            self.board.insert_marker(stim)
            im.image = self.stim_images[stim]
            stim_win.flip()
            core.wait(self.stim_time)

            #Empty the ring buffer into our data.
            self.eeg_data = np.concatenate((self.eeg_data, self.board.get_board_data()), axis=1)

        #Display the non-stim for a final wait time.
        im.image = self.stim_images[0]
        stim_win.flip()
        core.wait(self.wait_time)

        #Grab the final bit of data and end the stream.
        self.eeg_data = np.concatenate((self.eeg_data, self.board.get_board_data()), axis=1)
        self.board.stop_stream()
        self.board.release_session()

        # Remove image and display final message
        im.setAutoDraw(False)
        stim_win.flip()
        msg.text = "Testing Run Complete..."
        msg.setAutoDraw(True)
        stim_win.flip()
        core.wait(3)
        stim_win.close()

        if return_raw:
            #Transform EEG Data to MNE Raw file.
            return self.eeg_to_raw()
        else:
            return

    def save_data(self, file_name, double=False):
        # Check to make sure we have data.
        if self.eeg_data is None:
            raise Exception("Error: No EEG data has been recorded.")

        #Append file type to name.
        file_name = file_name + '_raw.fif'

        #Change the data to raw and save.
        data = self.eeg_to_raw()
        if double:
            #This should be used if saving/loading the raw causes precision issues.
            data.save(fname=file_name, fmt=double)
        else:
            data.save(fname=file_name)
        return

