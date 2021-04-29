#This contains the classification methods that can be considered 'deep neural networks' or 'deep learning'.
#Primarily involves the use of Keras/Theano to generate classifiers.
#Content may be based on/influenced by:

import tensorflow
from keras.callbacks import TensorBoard
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout, Input, Activation
from keras.layers import Flatten, Conv2D, MaxPooling2D, LSTM
from keras.layers import BatchNormalization, Conv3D, MaxPooling3D
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold


def feedforward_nn(data_shape, n_classes=2, model_shape=[64, 32, 16, 8], d_rate=0.2, b_norm=False):
    #Our shape should not include the batches, which
    #is the 0th element of the array model_shape
    shape = data_shape[1:]
    model = Sequential(name="FeedForwardNeuralNetwork")

    #For a dense feed-forward network, we need to flatten the data.
    model.add(Flatten(input_shape=shape))

    #Now, for each part of model_shape, we add a dense layer and activation.
    for layer_size in model_shape:
        model.add(Dense(layer_size))
        if b_norm:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(rate=d_rate))

    #Finally, add one last Dense layer with nodes equal to our class output.
    model.add(Dense(n_classes, activation='softmax'))
    # Now we compile and return the model.
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
               epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics='accuracy')
    model.summary()
    return model, opt

def convolutional_nn(data_shape, cnn_shape=[64, 64], dense_shape=[4096, 2048], n_classes=2, filt_size=3, pool_size=2, d_rate=0.5):
    shape = data_shape[1:]
    if len(cnn_shape) < 2:
        print('Warning: Need at least two layers for CNN')
    model = Sequential()
    model.add(Conv2D(cnn_shape[0], filt_size,
                     input_shape=shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    #Add Convolutional Layers
    if len(cnn_shape) > 2:
        for size in cnn_shape[1:-1]:
            model.add(Conv2D(size, filt_size, padding='same'))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Flatten())

    #Add dense layers
    if len(dense_shape) < 1:
        print('Warning: Need at least one dense layer')
    for size in dense_shape:
        model.add(Dense(size))
        model.add(Activation('relu'))
        model.add(Dropout(d_rate))

    #Final Layers
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
               epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics='accuracy')
    model.summary()
    return model, opt

def lstm_nn(data_shape, model_shape=[16, 8, 4], n_classes=2):
    return

def perform_crossval(X, Y, model, n_splits, epochs, batch_size, binary=True, n_classes=2, opt=Adam()):
    print("Performing cross-validation on model: " + model.name)
    #Create our K-Folds. Use shuffle, as the data is not randomised.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    #This will hold our accuracy values
    accuracy=[]
    loss=[]

    #For each split, compile a new model, train it, and test it.
    for train_index, test_index in skf.split(X, Y):
        #Copy our model.
        test_model = clone_model(model)
        test_model.compile(optimizer=opt, loss='categorical_crossentropy',
                           metrics='accuracy')
        test_model.summary()

        #Train the model
        test_model.fit(X[train_index], to_categorical(Y[train_index], n_classes), epochs=epochs, batch_size=batch_size,
                       validation_split=0.2, shuffle=True)


        #Test the model
        results = test_model.evaluate(X[test_index], to_categorical(Y[test_index], n_classes))
        results = dict(zip(test_model.metrics_names, results))

        #And get the accuracy
        accuracy.append((results['accuracy']))
        loss.append((results['loss']))

    #Print our results
    print("Cross-Validated Accuracy for %f splits: %f" % (n_splits, np.mean(accuracy)))
    print("Cross-Validated Loss for %f splits: %f" % (n_splits, np.mean(loss)))
    print(accuracy)
    print(loss)
    return