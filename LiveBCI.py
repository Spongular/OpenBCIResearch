#This script contains all the methods needed for live BCI recording, visualisation and decoding.

#System Imports
from random import shuffle

#Brainflow Imports
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

#Assorted Math/Numerical Imports
import numpy as np

#MNE EEG Decoding Imports
import mne
from mne import pick_types, Epochs
from mne import io

#Machine Learning Imports
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle

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
        if len(ch_names) < 1:
            print("No channel names provided. Using generic names")
            for x in range(0, ch_count):
                self.ch_names.append("CH{num}".format(num=x))

        #And append our stim channel name.
        self.ch_names.append("STI001")

        #Classifier Attributes
        self.classifier = classifier
        self.classifier_type = classifier_type

        #Check the type to ensure it's correct.
        if self.classifier_type != 'sklearn' and self.classifier_type != 'keras'\
                and self.classifier_type is not None:
            raise Exception("Error: classifier_type must be either 'sklearn', 'keras' or None")

        #IO and Visualising Attributes
        self.output_location = output_location
        self.visualize = visualize
        self.eeg_data = None
        self.raw = None
        self.epochs = None
        self.evoked = None

    #---------------------------Setting Attributes---------------------------------------------------------------------#

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

    #---------------------------Data Manipulation/Clearing-------------------------------------------------------------#

    def eeg_to_raw(self):
        #Check to make sure we have data.
        if self.eeg_data is None:
            raise Exception("Error: No EEG data has been recorded.")

        #Channel Count is used to determine what data is used.
        ch_count = len(self.ch_names) - 1
        #Form an MNE Raw object from our data.
        eeg_channels = self.board.get_eeg_channels(self.board.board_id)
        eeg_channels = eeg_channels[:ch_count]
        eeg_data = self.eeg_data[eeg_channels, :]
        eeg_data = eeg_data / 1000000  # BrainFlow returns uV, convert to V for MNE
        # Add the marker/stim channel to the data.
        eeg_data = np.concatenate((eeg_data, self.eeg_data[[self.board.get_marker_channel(self.board.board_id)], :]),
                                  axis=0)

        # Creating MNE objects from brainflow data arrays
        ch_types = ['eeg'] * ch_count
        # Need to append 'stim' info.
        ch_types.append('stim')
        sfreq = self.board.get_sampling_rate(self.board.board_id)
        info = mne.create_info(ch_names=self.ch_names, sfreq=sfreq, ch_types=ch_types)
        self.raw = mne.io.RawArray(eeg_data, info)

        return self.raw

    def filter_raw(self, min=4., max=None, notch_val=50.):
        #Check that we have data.
        if self.raw is None:
            raise Exception("Error: There is no raw data to filter.")

        print("\nNotch Filtering at {nv}hz...".format(nv=notch_val))
        self.raw.notch_filter(notch_val, filter_length='auto', phase='zero')

        if max is None:
            print("\nPerforming highpass filter from {l}Hz...".format(l=min))
            self.raw.filter(min, max, fir_design='firwin', skip_by_annotation=('edge', 'bad_acq_skip'))
        elif max > min:
            print("\nPerforming bandpass filter on range {l}Hz to {h}Hz...".format(l=min, h=max))
            self.raw.filter(min, max, fir_design='firwin', skip_by_annotation=('edge', 'bad_acq_skip'))
        else:
            raise Exception("Error: Attribute \'max\' is invalid. Specify either \'None\' or a value greater than "
                            "\'min\'.")

        return self.raw

    #The attribute recalc_data is used when new data has been recorded, but old raw is still recorded,
    #and not properly removed.
    def eeg_to_epochs(self, tmin, tmax, event_dict=dict(T1=2, T2=3), preprocess='highpass',
                      recalc_data=False, eeg_reject_uV=None, plot_bads=False, stim_ch='STI001'):
        #If we have no data at all...
        if self.eeg_data is None and self.raw is None:
            raise Exception("Error: No eeg data is present.")
        #Otherwise, if we have data but need new raw...
        if recalc_data or self.raw is None:
            self.eeg_to_raw()

        print("\nGathering Events...")
        #Now, we form the epochs.
        events = mne.find_events(raw=self.raw, stim_channel=stim_ch)
        print("\nEvents:")
        print(events)

        print('\nSetting Picks...')
        picks = pick_types(self.raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

        print('\nForming Epochs...')
        # Now, if we have no reject, we perform it normally, otherwise we drop epochs based on the threshold.
        if eeg_reject_uV is None:
            epochs = Epochs(self.raw, events, event_dict, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)
        else:
            epochs = Epochs(self.raw, events, event_dict, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True, reject=dict(eeg=eeg_reject_uV * 1e-6))

        print('\nDropping Bads...')
        epochs.drop_bad()
        print(epochs.drop_log)

        # Plot our bads.
        if plot_bads:
            epochs.plot_drop_log()

        #Set epochs and return
        self.epochs = epochs
        return self.epochs

    #Resamples the epochs to a specific rate. As resampling can cause edge artifacts, it's recommended
    #that the epochs be longer on either edge than required, and then cropped to avoid the issue.
    def resample_epochs(self, new_rate=160):
        #Make sure we have epoch data.
        if self.epochs is None:
            raise Exception("Error: No epoch data is present.")

        #Resample.
        print('Sampling rate before: {sfreq}Hz'.format(sfreq=self.epochs.info['sfreq']))
        self.epochs = self.epochs.resample(new_rate)
        print('Sampling rate after: {sfreq}Hz'.format(sfreq=self.epochs.info['sfreq']))

        return self.epochs

    def epochs_to_evoked(self):
        # If we have no data at all...
        if self.eeg_data is None and self.raw is None and self.epochs is None:
            raise Exception("Error: No eeg data is present.")
        elif self.epochs is None:
            raise Exception("Error: No epoch data is present.")

        #Average the epochs to get the evoke data.
        self.evoked = self.epochs.average()
        return self.evoked

    def clear_raw(self):
        self.raw = None
        return

    def clear_epochs(self):
        self.epochs = None
        return

    def clear_evoked(self):
        self.evoked = None
        return

    #---------------------------Stimulation Tests----------------------------------------------------------------------#

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
            self.board.insert_marker(1)
            core.wait(self.wait_time)

            #Set stim image, marker and wait.
            stim = stim_bag.pop()
            self.board.insert_marker(stim + 1)
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

    #---------------------------Classification Tools-------------------------------------------------------------------#

    def __get_data_labels(self, event_bounds=[0,-1], scale=None):
        #Check the epochs.
        if self.epochs is None:
            raise Exception("Error: No epoch data found.")

        #Grab the data
        if scale is not None:
            data = self.epochs.get_data() * scale
        else:
            data = self.epochs.get_data()

        #Get the labels
        labels = self.epochs.events[event_bounds[0], event_bounds[1]]

        return data, labels

    #Accepts a Numpy Array and expands it.
    def __expand_data(self, data, dims=3):
        if dims <= data.ndim:
            raise Exception("Error: Cannot raise dimensions below or equal to current dimension count.")
        data = np.expand_dims(data, 3)
        return data

    #Accepts a list of label sets and one-hot encodes them to the number of categories there are.
    def __one_hot_encode(self, label_sets=[], categories=2):
        for x in range(0, len(label_sets)):
            label_sets[x] = to_categorical(label_sets[x], categories)
        return label_sets


    def __classify(self, scale=None, num_classes=2, class_weights=None):
        #Grab our data
        if scale is None:
            X = self.epochs.get_data()
        else:
            X = self.epochs.get_data() * scale
        y = self.epochs.events[:, -1] - 2 #Assume that T0 is rest, as it should be.

        #Shuffle the data to avoid issues.
        X, y = shuffle(X, y, random_state=1)

        if self.classifier_type == 'keras':
            #For keras, we grab our model's expected input and change our data to match.
            input_shape = self.classifier.get_config()["layers"][0]["config"]["batch_input_shape"]
            # Make sure that the values we want match. i.e. input if input is (None, 128, 96, 1), our data must be
            # at least (num_trials, 128, 96, 1 or None).
            if input_shape[1] != X.shape[1] or input_shape[2] != X.shape[2]:
                raise Exception("Error: Data shape {data_shape} is not ".format(data_shape=X.shape) +
                                "compatible with input shape of {input_shape}.".format(input_shape=input_shape))
            if len(input_shape) > len(X.shape):
                X = self.__expand_data(X, dims = 3)
            elif len(input_shape) == len(X.shape) and len(input_shape) > 3:
                if input_shape[3] != X.shape[3]:
                    raise Exception("Error: Data shape {data_shape} is not ".format(data_shape=X.shape) +
                                "compatible with input shape of {input_shape}.".format(input_shape=input_shape))
            #Keras needs one-hot encoded data for labels.
            y = self.__one_hot_encode(y, num_classes)
            if class_weights is None:
                class_weights = {0:1, 1:1}
        elif self.classifier_type == 'sklearn':
            return #Put some stuff here when I care.

        #This function is universal between Keras and Sk-learn
        result = self.classifier.predict(X)

        #Return X and y for metric calculations.
        return result, X, y

    def set_keras_classifier(self, classifier):
        self.classifier = classifier
        self.classifier_type = 'keras'
        return

    def set_sklearn_classifier(self, classifier):
        self.classifier = classifier
        self.classifier_type = 'sklearn'
        return

    def load_keras_classifier(self, path, type='model'):
        if type == 'model':
            #In this instance, we just make a new model from the file.
            self.classifier = load_model(path)
            self.classifier_type = 'keras'
        elif type == 'weights':
            #We need a model to just set weights
            if self.classifier is None:
                raise Exception("Error: Loading keras weights requires a model.")
            if self.classifier_type != 'keras':
                raise Exception("Error: Existing classifier object is not a keras model.")
            #Here, we're assuming that we have a valid model that fits the weights specified in path.
            #If we don't, well, too bad.
            self.classifier.load_weights(path)
        else:
            raise Exception("Error: Invalid type \'{t}\' specified.".format(t=type))
        return

    def load_sklearn_classifier(self, path):
        return


    #---------------------------Offline Classification-----------------------------------------------------------------#

    def offline_classification(self, metrics=['accuracy'], data_scale=None):
        #Check for data
        if self.epochs is None:
            if self.raw is None:
                raise Exception("Error: No epochs nor raw eeg data present. Please record or load data.")
            raise Exception("Error: No epochs present. Please process raw eeg into epochs with eeg_to_epochs")

        #Depending on the method, we'll perform our classification.
        print("\nPerforming classification...")
        result, X, y = self.__classify(scale=data_scale, num_classes=(len(self.stim_images) - 1))

        if 'accuracy' in metrics:
            #Need to differentiate as Keras needs one-hot encoded labels,
            #but sklearn doesn't, thus there are different methods.
            if self.classifier_type == 'keras':
                acc = np.mean(result.argmax(axis=-1) == y.argmax(axis=-1))
                print("\nAccuracy: {accuracy}".format(accuracy=acc))
            elif self.classifier_type == 'sklearn':
                #insert something here.
                return
        #if ''
        return

    #---------------------------Online Classification------------------------------------------------------------------#


    #---------------------------Data Saving/Loading--------------------------------------------------------------------#

    def save_data(self, file_name, double=False, type='imagery'):
        # Check to make sure we have data.
        if self.eeg_data is None:
            raise Exception("Error: No EEG data has been recorded.")
        if type != 'imagery' and type != 'movement':
            raise Exception("Error: Parameter 'type' must be string 'imagery' or 'movement'")

        # Format the file name to include format and directory.
        if type == 'imagery':
            file_name = 'LiveRecordings\\MotorResponses\\Imagery\\' + file_name + '_raw.fif'
        elif type == 'movement':
            file_name = 'LiveRecordings\\MotorResponses\\Movement\\' + file_name + '_raw.fif'

        #Change the data to raw and save.
        data = self.eeg_to_raw()
        if double:
            #This should be used if saving/loading the raw causes precision issues.
            data.save(fname=file_name, fmt=double)
        else:
            data.save(fname=file_name)
        return

    def load_data(self, file, format='fif', overwrite=False):
        #Deal with the situation where there is data already.
        if self.raw is not None and overwrite == False:
            raise Exception('Error: Data already present. Please set overwrite=True to load new data.')

        if format == 'fif':
            self.raw = io.read_raw_fif(fname=file, preload=True)
        elif format == 'edf':
            self.raw = io.read_raw_edf(fname=file, preload=True)
        else:
            raise Exception('Error: Invalid file type. please specify \'fif\' or \'edf\' for attribute \'format\'.')

        return self.raw

    def load_multiple_data(self, files=[], format='fif', overwrite=False):
        #Deal with the situation where there is data already.
        if self.raw is not None and overwrite == False:
            raise Exception('Error: Data already present. Please set overwrite=True to load new data.')

        if format == 'fif':
            self.raw = io.concatenate_raws([io.read_raw_fif(f, preload=True) for f in files])
        elif format == 'edf':
            self.raw = io.concatenate_raws([io.read_raw_edf(f, preload=True) for f in files])
        else:
            raise Exception('Error: Invalid file type. please specify \'fif\' or \'edf\' for attribute \'format\'.')

        return self.raw
