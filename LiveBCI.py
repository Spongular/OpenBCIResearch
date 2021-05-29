#This script contains all the methods needed for live BCI recording, visualisation and decoding.

#System Imports
import time

#Brainflow Imports
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

#Assorted Math/Numerical Imports
import matplotlib
import numpy as np

#MNE EEG Decoding Imports

#Machine Learning Imports

#PsychoPi EEG Stimulation Imports

def generateSyntheticBoard():
    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    return board

def generateOpenBCIBoard():
    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    return board

#This is the stimulator for motor imagery.
class MotorImageryStimulator:
    def __init__(self, stim_time, wait_time, stim_count, stim_images=[], stim_image_size=[600,600], stim_type='lr',
                 board=None, ch_count=4, ch_names=[], classifier=None, classifier_type=None, output_location='DEFAULT',
                 visualize=False):
        #Stimulation Attributes
        self.stim_time = stim_time
        self.wait_time = wait_time
        self.stim_count = stim_count
        self.stim_names
        self.stim_images

        #If we are not passed a list of image locations, we use the default.
        if stim_images == []:
            self.stim_images = [] #This will hold the strings for the location of the stim images.
            self.stim_images.append('INSERT_NON_STIM_IMAGE_LOCATION') #This is our 'blank canvas'
            self.stim_names.append('WAIT')
            if stim_type == 'lr':
                self.stim_images.append('INSERT_LEFT_HAND_STIM_IMAGE_LOCATION')
                self.stim_names.append('LEFT')
                self.stim_images.append('INSERT_RIGHT_HAND_STIM_IMAGE_LOCATION')
                self.stim_names.append('RIGHT')
            elif stim_type == 'hf':
                self.stim_images.append('INSERT_HANDS_STIM_IMAGE_LOCATION')
                self.stim_names.append('HANDS')
                self.stim_images.append('INSERT_FEET_STIM_IMAGE_LOCATION')
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

    #This is used to test the stimulation window.
    #Does not read or record from board.
    def test_stim(self):
        return

    #This is the primary function for recording
    #eeg data. Presents the stimulation and records the data
    #to a mne raw object to return
    def run_stim(self):
        return


