import gen_tools
import data_loading
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint
import keras_classifiers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


#Grab all the data and epoch.
tmin, tmax = -0., 4
#raw = data_loading.get_single_mi('080', 2)
raw = data_loading.get_all_mi_between(1, 81, 2, ["088", "092", "100"])
raw = gen_tools.preprocess_highpass(raw, 2)
event_id = dict(T1=0, T2=1)
X, y, epochs = gen_tools.epoch_data(raw, tmin, tmax, eeg_reject_uV=None, scale=None, pick_list=["C3", "Cz", "C4"])

#Perform Wavelet Transform for each event.
data, labels = gen_tools.wavelet_transform_general(epochs, event_names=epochs.event_id,
                                           f_low=4., f_high=30., f_num=10, shuffle=True,
                                           transform_type='multitaper')

#Normalise and then reshape from [Epochs, Channels, FreqNums, Data] to [Epochs, Channels*FreqNums, Data]
data = gen_tools.normalise(data)
#data = data * 1000
print("Before Reshape: {}".format(data.shape))
data = np.reshape(data, (data.shape[0], np.prod(data.shape[1:3]), data.shape[3]))
data = np.expand_dims(data, 3)
print("After Reshape: {}".format(data.shape))
np.prod(data.shape[1:2])

#Form the model.
model = keras_classifiers.EEGNet(nb_classes=2, Chans=data.shape[1], Samples=data.shape[2], ThirdAxis=data.shape[3],
               dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
               dropoutType='Dropout')
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                               save_best_only=True)
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

#Make data sets
data, labels = shuffle(data, labels, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
y_train = to_categorical(y_train, 2)
y_val = to_categorical(y_val, 2)
y_test = to_categorical(y_test, 2)

class_weights = {0:1, 1:1}

fittedModel = model.fit(X_train, y_train, batch_size = 5, epochs = 300,
                        verbose = 2, validation_data=(X_val, y_val),
                        class_weight=class_weights, callbacks=[checkpointer])

probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)
acc         = np.mean(preds == y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))
