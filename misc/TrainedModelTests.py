
import mne
from mne.io import read_raw_edf

import data_loading
import gen_tools
from sklearn.utils import shuffle
import numpy as np
from keras.utils import to_categorical
from keras import models
import os
import LiveBCI

def select_nn(path='E:/PycharmProjects/OpenBCIResearch/NN_Weights/convEEGNet/4-channel-headband/'):
    nn_list = os.listdir(path)

    count = 0
    for name in nn_list:
        print("{count} - {name}".format(count=count, name=name))
        count = count + 1
    del count

    user_input = ''
    while user_input.isdigit() == False:
        user_input = input("Select Network to Load...")

    user_input = int(user_input)
    if user_input < 0 and user_input >= nn_list.count():
        raise Exception("Error: Invalid Selection")

    print('Network no. {nn_num}, \'{nn_name}\' has been chosen'.format(nn_num=user_input, nn_name=nn_list[user_input]))

    model_path = path + nn_list[user_input]
    return model_path

def test_all_models(data, labels,
                    path='E:/PycharmProjects/OpenBCIResearch/NN_Weights/convEEGNet/4-channel-headband/',
                    ignore_criteria=['readme']):
    #Grab the list of instances and remove any with the 'ignore_criteria' string in it.
    nn_list = os.listdir(path)
    for file in nn_list:
        for ign in ignore_criteria:
            if file.find(ign) != -1:
                print('Removing file \'{file}\' from list'.format(file=file))
                nn_list.remove(file)
                break

    #Now, for each model, load it and test with the data. Keep track of the best result.
    best_nn_name = None
    best_nn_score = None
    for file in nn_list:
        print("Testing NN: {file}".format(file=file))
        model_path = path + file
        nn_model = models.load_model(model_path)
        probs = nn_model.predict(data)
        preds = probs.argmax(axis=-1)
        acc = np.mean(preds == labels.argmax(axis=-1))
        print("{file} accuracy: {acc} ".format(file=file, acc=acc))
        if best_nn_score == None or best_nn_score < acc:
            best_nn_name = file
            best_nn_score = acc
            print("New highest score of {best} for NN {best_nn}".format(best=best_nn_score, best_nn=best_nn_name))

    #And return our best scores
    return best_nn_name, best_nn_score

def retrain_all_models():
    return


#model_path = select_nn()
#nn_model = models.load_model(model_path)
#nn_model.summary()

#Now we load the recordings
files = ['E:\\PycharmProjects\\OpenBCIResearch\\DataGathering\\LiveRecordings\\MotorResponses\\Movement\\subject1-10_06_2021-mm-lr-001_raw.fif',
         'E:\\PycharmProjects\\OpenBCIResearch\\DataGathering\\LiveRecordings\\MotorResponses\\Movement\\subject1-10_06_2021-mm-lr-002_raw.fif',
         'E:\\PycharmProjects\\OpenBCIResearch\\DataGathering\\LiveRecordings\\MotorResponses\\Movement\\subject1-10_06_2021-mm-lr-003_raw.fif',
         'E:\\PycharmProjects\\OpenBCIResearch\\DataGathering\\LiveRecordings\\MotorResponses\\Movement\\subject1-10_06_2021-mm-lr-004_raw.fif']

board = LiveBCI.generateOpenBCIBoard('COM3', 'ganglion', timeout=15)
test = LiveBCI.MotorImageryStimulator(stim_time=4, wait_time=4, stim_count=5, stim_type='lr', board=board)
test.load_multiple_data(files=files)

#Then, we filter, epoch and resample to make the data match.
test.filter_raw(min=4.)
test.eeg_to_epochs(tmin=0., tmax=3., event_dict=dict(T1=2, T2=3), stim_ch='STI001').plot(block=True)
test.resample_epochs(new_rate=160)

epochs = test.epochs

#Grab our labels.
print(epochs.event_id)
labels = epochs.events[:, -1] - 2

#Scale the data
data = epochs.get_data() * 1000

#Shuffle and format the data to test.
data, labels = shuffle(data, labels, random_state=1)
print("Data Set Before Reshape: {}".format(data.shape))
print("Reshaping Data...")
data = gen_tools.reshape_3to4(data)
print("Data Set After Reshape: {}".format(data.shape))
labels = to_categorical(labels, 2)

best_name, best_score = test_all_models(data, labels)

print("\nOverall Best Accuracy: {score} from {name}".format(score=best_score, name=best_name))
# #Run the model
# probs       = nn_model.predict(data)
# preds       = probs.argmax(axis = -1)
# acc         = np.mean(preds == labels.argmax(axis=-1))
# print("Classification accuracy: %f " % (acc))