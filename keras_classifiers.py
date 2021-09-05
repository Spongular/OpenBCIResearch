#This contains the classification methods that can be considered 'deep neural networks' or 'deep learning'.
#Primarily involves the use of Keras/Theano to generate classifiers.
#Content may be based on/influenced by:
import sys

import tensorflow
from keras.callbacks import TensorBoard
import numpy as np
from keras.models import Sequential, clone_model, Model
from keras.layers import Dense, Dropout, Input, Activation, SpatialDropout2D, DepthwiseConv2D, AveragePooling2D
from keras.layers import Flatten, Conv2D, MaxPooling2D, LSTM, SeparableConv2D
from keras.layers import BatchNormalization, Conv3D, MaxPooling3D, Permute, concatenate
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
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
               epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics='accuracy')
    model.summary()
    return model, opt

def convolutional_nn(data_shape, cnn_shape=[50, 75, 120], n_classes=2,
                     pool_size=2, d_rate=0.25, b_norm=True, learn_r=0.001):
    shape = data_shape[1:]
    if len(cnn_shape) < 2:
        print('Warning: Need at least two layers for CNN')
    model = Sequential()
    model.add(Conv2D(cnn_shape[0], kernel_size=(1, 5),
                     input_shape=shape, padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=pool_size, strides=(1, 3), padding='same'))
    model.add(Dropout(d_rate))

    #Add Convolutional Layers
    if len(cnn_shape) > 2:
        for size in cnn_shape[1:]:
            model.add(Conv2D(size, kernel_size=(1, 5), padding='same'))
            if b_norm:
                model.add(BatchNormalization())
            model.add(Activation('elu'))
            model.add(AveragePooling2D(pool_size=pool_size, strides=(1, 3), padding='same'))
            model.add(Dropout(d_rate))

    # Final Layers
    model.add(Flatten())
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    opt = Adam(lr=learn_r, beta_1=0.9, beta_2=0.999,
               epsilon=None, decay=0.0, amsgrad=False)
    model.summary()
    return model, opt

def lstm_nn(data_shape, model_shape=[16, 8, 4], n_classes=2):

    return

#This is my own construction of EEGNet from the paper:
#An Accurate EEGNet-based Motor-Imagery Brain–Computer Interface for Low-Power Edge Computing
#First_tf_size is the Nf value form the paper, and pooling_length is the Np value.
#Expects data in shape (trial, samples, channels, 1), so use gen_tools.reshape_3to4()
#This link is useful to understand how they properly perform 1d depthwise:
#https://datascience.stackexchange.com/questions/93569/performing-1d-depthwise-conv-using-keras-2d-depthwise-conv
def convEEGNet(input_shape, chan, n_classes=2, n_filt = [8, 16, 16],
               first_tf_size=128, pooling_length=8, l_rate=0.01,
               d_rate=0.25):
    #Initialising.
    input_shape = input_shape[1:]
    model = Sequential(name="EEGNet-WangXiangEtAl")

    #First Layer.
    #This is the temporal convolution.
    model.add(Conv2D(filters=n_filt[0], kernel_size=[1, first_tf_size],
                     strides=(1, 1), padding='same', input_shape=input_shape
                     , use_bias=False))
    model.add(BatchNormalization(name="bnorm1")) #Take a look at the axis if this doesn't work.

    #Second Layer.
    #This is the depthwise convolution. We need to reorder the input to perform 1d depthwise.
    model.add(DepthwiseConv2D(depth_multiplier=2, kernel_size=[chan, 1],
                              padding='valid', use_bias=False, depthwise_constraint=max_norm(1.)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(AveragePooling2D(pool_size=(1, pooling_length),
                               padding='valid'))
    model.add(Dropout(d_rate))

    #Third Layer.
    #This is the separable convolution.
    model.add(SeparableConv2D(filters=n_filt[2], kernel_size=(1, 16), strides=(1, 1),
                              padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(AveragePooling2D(pool_size=(1, pooling_length), padding='valid'))
    model.add(Dropout(d_rate))

    #Fourth Layer.
    #This is the fully connected layer, where we flatten the data and
    #use a dense layer to condense our results to the classes.
    model.add(Flatten())
    model.add(Dense(n_classes, kernel_constraint=max_norm(0.25)))
    model.add(Activation('softmax'))

    opt = Adam(lr=l_rate)
    model.summary()
    return model, opt

def test(nb_classes, Chans=64, Samples=128, ThirdAxis=1,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, l_rate=0.001):

    model = Sequential()

    ##################################################################
    model.add(Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, ThirdAxis),
                    use_bias=False))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(AveragePooling2D((1, 4)))
    model.add(Dropout(dropoutRate))

    model.add(SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(AveragePooling2D((1, 8)))
    model.add(Dropout(dropoutRate))

    model.add(Flatten(name='flatten'))

    model.add(Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate)))
    model.add(Activation('softmax', name='softmax'))
    opt = Adam(lr=l_rate)
    model.summary()
    return model, opt

def EEGNetScheduler(epoch, lr):
    #Our Values
    lr_vals = [0.01, 0.001, 0.0001]
    if epoch < 20:
        return lr_vals[0]
    elif epoch < 50:
        return lr_vals[1]
    else:
        return lr_vals[2]

def perform_crossval(X, Y, model, n_splits, epochs, batch_size, n_classes=2, opt=Adam()):
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
#Altered it to accept input [Chans, Samples, ThirdAxis] rather than [Chans, Samples, 1]
def EEGNet(nb_classes, Chans=64, Samples=128, ThirdAxis=1,
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

    input1 = Input(shape=(Chans, Samples, ThirdAxis))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, ThirdAxis),
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

    return Model(inputs=input1, outputs=softmax, name="EEGNet-MilResearch")


def DeepConvNet(nb_classes, Chans=64, Samples=256,
                dropoutRate=0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.

    This implementation assumes the input is a 2-second EEG signal sampled at
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10

    Note that this implementation has not been verified by the original
    authors.

    """

    # start the model
    input_main = Input((Chans, Samples, 1))
    block1 = Conv2D(25, (1, 5),
                    input_shape=(Chans, Samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(25, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

#This is based on a combination of the work found at: https://github.com/rootskar/EEGMotorImagery/blob/master/EEGModels.py
#And the slightly modified EEGNet implementation found above.
#This is just changed up to match the proper format for input data, and to have the kernels match the model
#in the paper by default, not the model in the provided git link.
#That model had: conv_kernels=(8, 16, 32) and sep_conv_kernels=(16, 32, 64)
def fusionEEGNet(n_classes, chans=64, samples=128, third_axis=1, l_rate=0.01,
                 dropout_rate=0.5, norm_rate=0.25, dropout_type='Dropout',
                 batch_norm=True, conv_kernels=(8, 16, 32), sep_conv_kernels=(16, 32, 64)):

    #Determine the dropout type.
    #There's no reason not to provide this option just like the EEGNet method.
    if dropout_type == 'SpatialDropout2D':
        dropout_type = SpatialDropout2D
    elif dropout_type == 'Dropout':
        dropout_type = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    #This input block will feed into all 3 'branches' of the classifier.
    input1 = Input(shape=(chans, samples, third_axis))

    #Branch 1
    B1 = Conv2D(conv_kernels[0], (1, 64), padding='same',
                  input_shape=(chans, samples, third_axis),
                  use_bias=False)(input1)
    if batch_norm:
        B1 = BatchNormalization()(B1)
    B1 = DepthwiseConv2D((chans, 1), use_bias=False,
                             depth_multiplier=2,
                             depthwise_constraint=max_norm(1.))(B1)
    if batch_norm:
        B1 = BatchNormalization()(B1)
    B1 = Activation('elu')(B1)
    B1 = AveragePooling2D((1, 4))(B1)
    if dropout_rate > 0:
        B1 = dropout_type(dropout_rate)(B1)
    B1 = SeparableConv2D(sep_conv_kernels[0], (1, 8),
                             use_bias=False, padding='same')(B1)
    if batch_norm:
        B1 = BatchNormalization()(B1)
    B1 = Activation('elu')(B1)
    B1 = AveragePooling2D((1, 8))(B1)
    if dropout_rate > 0:
        B1 = dropout_type(dropout_rate)(B1)
    B1 = Flatten(name='Flatten1')(B1)

    #Branch 2
    B2 = Conv2D(conv_kernels[1], (1, 128), padding='same',
                  input_shape=(chans, samples, third_axis),
                  use_bias=False)(input1)
    if batch_norm:
        B2 = BatchNormalization()(B2)
    B2 = DepthwiseConv2D((chans, 1), use_bias=False,
                         depth_multiplier=2,
                         depthwise_constraint=max_norm(1.))(B2)
    if batch_norm:
        B2 = BatchNormalization()(B2)
    B2 = Activation('elu')(B2)
    B2 = AveragePooling2D((1, 4))(B2)
    if dropout_rate > 0:
        B2 = dropout_type(dropout_rate)(B2)
    B2 = SeparableConv2D(sep_conv_kernels[1], (1, 16),
                         use_bias=False, padding='same')(B2)
    if batch_norm:
        B2 = BatchNormalization()(B2)
    B2 = Activation('elu')(B2)
    B2 = AveragePooling2D((1, 8))(B2)
    if dropout_rate > 0:
        B2 = dropout_type(dropout_rate)(B2)
    B2 = Flatten(name='Flatten2')(B2)

    #Branch 3
    B3 = Conv2D(conv_kernels[2], (1, 256), padding='same',
                  input_shape=(chans, samples, third_axis),
                  use_bias=False)(input1)
    if batch_norm:
        B3 = BatchNormalization()(B3)
    B3 = DepthwiseConv2D((chans, 1), use_bias=False,
                         depth_multiplier=2,
                         depthwise_constraint=max_norm(1.))(B3)
    if batch_norm:
        B3 = BatchNormalization()(B3)
    B3 = Activation('elu')(B3)
    B3 = AveragePooling2D((1, 4))(B3)
    if dropout_rate > 0:
        B3 = dropout_type(dropout_rate)(B3)
    B3 = SeparableConv2D(sep_conv_kernels[2], (1, 32),
                         use_bias=False, padding='same')(B3)
    if batch_norm:
        B3 = BatchNormalization()(B3)
    B3 = Activation('elu')(B3)
    B3 = AveragePooling2D((1, 8))(B3)
    if dropout_rate > 0:
        B3 = dropout_type(dropout_rate)(B3)
    B3 = Flatten(name='Flatten3')(B3)

    #Merging
    merge = concatenate([B1, B2])
    merge = concatenate([merge, B3])

    #Final Dense Layer / Misc
    flatten = Flatten(name='Flatten4')(merge)
    dense = Dense(n_classes, name='Dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='Softmax')(dense)

    model = Model(inputs=input1, outputs=softmax, name="FusionEEGNet")
    opt = Adam(lr=l_rate)
    model.summary()
    return model, opt
