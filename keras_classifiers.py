#This contains the classification methods that can be considered 'deep neural networks' or 'deep learning'.
#Primarily involves the use of Keras/Theano to generate classifiers.
#Content may be based on/influenced by:

import tensorflow
from keras.callbacks import TensorBoard
import numpy as np
from keras.models import Sequential, clone_model, Model
from keras.layers import Dense, Dropout, Input, Activation, SpatialDropout2D, DepthwiseConv2D, AveragePooling2D
from keras.layers import Flatten, Conv2D, MaxPooling2D, LSTM, SeparableConv2D
from keras.layers import BatchNormalization, Conv3D, MaxPooling3D
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.constraints import max_norm
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

#NOT MINE, FROM https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
def EEGNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)