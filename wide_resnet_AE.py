import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage

import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Activation, Flatten, Lambda, Convolution2D, AveragePooling2D, BatchNormalization, Dropout
from keras.engine import Input, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import merge
from keras import initializers, layers
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras.backend as K
import json
import time
import os

def save_history(history, result_file,epochs):
    loss = history.history['loss']
    #conv_loss = history.history['conv_out_loss']
    acc = history.history['conv_out_acc']
    cat_acc = history.history['softmax_acc']
    val_loss = history.history['val_loss']
    #val_conv_loss = history.history['val_out_recon_loss']
    val_acc = history.history['val_conv_out_acc']
    val_cat_acc = history.history['val_softmax_acc']
    nb_epoch = len(acc)

    with open(result_file, "a") as fp:
        if epochs==0:
            fp.write("i\tloss\tconv_acc\tcat_acc\tval_loss\tval_conv_acc\tval_cat_acc\n")
            for i in range(nb_epoch):
                fp.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epochs, loss[i], acc[i], cat_acc[i], val_loss[i], val_acc[i], val_cat_acc[i]))
        else:
            for i in range(nb_epoch):
                fp.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epochs, loss[i], acc[i], cat_acc[i], val_loss[i], val_acc[i], val_cat_acc[i]))

# Learning rate schedule
def lr_sch(epoch):
    if epoch < 2:
        return 0.001  #0.1
    elif epoch < 4:  #120
        return 0.0005 #0.02
    elif epoch < 8:  #160
        return 0.00001  #0.004
    else:
        return 0.00001  #0.0008

# Learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_sch)
                
def train(adm, model, data, epoch_size=32,batch_size=128):
    (x_train, y_train), (x_test, y_test) = data
    # Learning rate schedule
    #loss='binary_crossentropy'  'categorical_crossentropy' 'mse'
    model.compile(optimizer=adm,
                  loss={'softmax':'categorical_crossentropy', 'conv_out':'mse'},
                  loss_weights=[1., 100.],
                  metrics={'softmax': 'accuracy','conv_out': 'accuracy'})

    history=model.fit([x_train, y_train],[y_train, x_train], batch_size=batch_size, epochs=epoch_size,
              validation_data=[[x_test, y_test], [y_test, x_test]],callbacks=[lr_scheduler])

    return model, history

def to3d(X):
    if X.shape[-1]==3: return X
    b = X.transpose(3,1,2,0)
    c = np.array([b[0],b[0],b[0]])
    return c.transpose(3,1,2,0)

def plot_generated_batch(i, model,data1,data2):
    x_test, y_test = data1
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=32)
    X_gen = x_recon
    X_raw = x_test   
    
    Xs1 = to3d(X_raw[:10])
    Xg1 = to3d(X_gen[:10])
    Xs1 = np.concatenate(Xs1, axis=1)
    Xg1 = np.concatenate(Xg1, axis=1)
    Xs2 = to3d(X_raw[10:20])
    Xg2 = to3d(X_gen[10:20])
    Xs2 = np.concatenate(Xs2, axis=1)
    Xg2 = np.concatenate(Xg2, axis=1)
    
    x_train, y_train = data2
    y_pred, x_recon = model.predict([x_train, y_train], batch_size=32)
    X_gen = x_recon
    X_raw = x_train   
    
    Xs3 = to3d(X_raw[:10])
    Xg3 = to3d(X_gen[:10])
    Xs3 = np.concatenate(Xs3, axis=1)
    Xg3 = np.concatenate(Xg3, axis=1)
    Xs4 = to3d(X_raw[10:20])
    Xg4 = to3d(X_gen[10:20])
    Xs4 = np.concatenate(Xs4, axis=1)
    Xg4 = np.concatenate(Xg4, axis=1)
    
    XX = np.concatenate((Xs1,Xg1,Xs2,Xg2,Xs3,Xg3,Xs4,Xg4), axis=0)
    plt.imshow(XX)
    plt.axis('off')
    plt.savefig("./caps_figures/WresnetACE{0:03d}.png".format(i))
    plt.pause(3)
    plt.close()

nb_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# reorder dimensions for tensorflow
x_train = np.transpose(x_train.astype('float32'), (0, 1, 2,3))
mean = np.mean(x_train, axis=0, keepdims=True)
std = np.std(x_train)
x_train = x_train/255.    #(x_train - mean) / std
x_test = np.transpose(x_test.astype('float32'), (0, 1,2, 3))
x_test = x_test/255.   #(x_test - mean) / std
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


def residual_block(x, nb_filters=16, subsample_factor=1):
    
    prev_nb_channels = K.int_shape(x)[3]

    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        shortcut = AveragePooling2D(pool_size=subsample, dim_ordering='tf')(x)
    else:
        subsample = (1, 1)
        # shortcut: identity
        shortcut = x
        
    if nb_filters > prev_nb_channels:
        
        shortcut = Lambda(zero_pad_channels,
                          arguments={'pad': nb_filters - prev_nb_channels})(shortcut)
        
    y = BatchNormalization(axis=3)(x)
    y = Activation('relu')(y)
    y = Convolution2D(nb_filters, 3, 3, subsample=subsample,
                      init='he_normal', border_mode='same', dim_ordering='tf')(y)
    y = BatchNormalization(axis=3)(y)
    y = Activation('relu')(y)
    y = Dropout(0.5)(y)
    y = Convolution2D(nb_filters, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(y)
    
    out = merge([y, shortcut], mode='sum')

    return out

#%%time

img_rows, img_cols = 32, 32
img_channels = 3

blocks_per_group = 4
widening_factor = 5  #10

input_shape=(img_rows, img_cols, img_channels)
#inputs = Input(shape=(img_rows, img_cols, img_channels))
input1 = layers.Input(shape=input_shape)
input2 = layers.Input(shape=(nb_classes,))


x = Convolution2D(16, 3, 3, 
                  init='he_normal', border_mode='same', dim_ordering='tf')(input1)

for i in range(0, blocks_per_group):
    nb_filters = 8 * widening_factor  #16
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=1)

x1 = residual_block(x, nb_filters=16 * widening_factor, subsample_factor=2)  
x = residual_block(x1, nb_filters=16 * widening_factor, subsample_factor=1)  
for i in range(2, blocks_per_group):
    nb_filters = 16 * widening_factor  #32
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

for i in range(0, blocks_per_group):
    nb_filters = 32 * widening_factor  #64
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = AveragePooling2D(pool_size=(8, 8), strides=None, border_mode='valid', dim_ordering='tf')(x)
x = Flatten()(x)

softmax = Dense(nb_classes, activation='softmax', name='softmax')(x)

conv_out=Convolution2D(32 * widening_factor, (3, 3),activation='relu', padding="same")(x1)
conv_out=UpSampling2D(size=(2, 2))(conv_out)
"""
conv_out=Convolution2D(32 * widening_factor, (3, 3),activation='relu', padding="same")(conv_out)
conv_out=UpSampling2D(size=(2, 2))(conv_out)
"""
conv_out=Convolution2D(3, (3, 3), padding="same",activation='sigmoid', name="conv_out")(conv_out)


model = Model([input1,input2], [softmax, conv_out])

model.summary()

#%%time


sgd = SGD(lr=0.1, decay=5e-4, momentum=0.9, nesterov=True)
adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)


batch_size = 128
nb_epoch = 100
data_augmentation = True

# Model saving callback
#checkpointer = ModelCheckpoint(filepath='stochastic_depth_cifar10.hdf5', verbose=1, save_best_only=True)

if not data_augmentation:
    print('Not using data augmentation.')
    
    #model.load_weights('params_WResnetAE_epoch_019.hdf5')
    for j in range(20):
        model, history = train(adm,model=model, data=((x_train, y_train), (x_test, y_test)), epoch_size=10 ,batch_size=batch_size)
        model.save_weights('params_WResnetAE_epoch_{0:03d}.hdf5'.format(j), True)
        plot_generated_batch(j,model=model, data1=(x_train, y_train),data2=(x_test, y_test))
        # 学習履歴を保存
        save_history(history, os.path.join("./caps_figures/", 'history_Wresnet_AE.txt'),j)
        
else:
    print('Using real-time data augmentation.')

    # realtime data augmentation
    datagen_train = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True,
        vertical_flip=False)
    datagen_train.fit(x_train)
    
    #model.load_weights('params_vggAE_epoch_011.hdf5')
    
    for j in range(20):
        model, history = train(adm,model=model, data=((x_train, y_train), (x_test, y_test)), epoch_size=10,batch_size=batch_size)
        model.save_weights('params_WresnetACE_epoch_{0:03d}.hdf5'.format(j), True)
        plot_generated_batch(j,model=model, data1=(x_train, y_train),data2=(x_test, y_test))
        # 学習履歴を保存
        save_history(history, os.path.join("./caps_figures/", 'history_Wresnet_ACE.txt'),j)
        
"""
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 32, 32, 3)    0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 16)   448         input_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32, 32, 16)   64          conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 32, 16)   0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 40)   5800        activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 32, 40)   160         conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 32, 32, 40)   0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 32, 32, 40)   0           activation_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 32, 40)   14440       dropout_1[0][0]
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 32, 32, 40)   0           conv2d_1[0][0]
__________________________________________________________________________________________________
merge_1 (Merge)                 (None, 32, 32, 40)   0           conv2d_3[0][0]
                                                                 lambda_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 32, 32, 40)   160         merge_1[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 32, 32, 40)   0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 40)   14440       activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 32, 32, 40)   160         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 32, 32, 40)   0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 32, 32, 40)   0           activation_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 40)   14440       dropout_2[0][0]
__________________________________________________________________________________________________
merge_2 (Merge)                 (None, 32, 32, 40)   0           conv2d_5[0][0]
                                                                 merge_1[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 32, 32, 40)   160         merge_2[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 32, 32, 40)   0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 32, 40)   14440       activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 32, 32, 40)   160         conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 32, 40)   0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 32, 32, 40)   0           activation_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 40)   14440       dropout_3[0][0]
__________________________________________________________________________________________________
merge_3 (Merge)                 (None, 32, 32, 40)   0           conv2d_7[0][0]
                                                                 merge_2[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 32, 32, 40)   160         merge_3[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 32, 32, 40)   0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 32, 40)   14440       activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 32, 32, 40)   160         conv2d_8[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 32, 32, 40)   0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 32, 32, 40)   0           activation_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 32, 32, 40)   14440       dropout_4[0][0]
__________________________________________________________________________________________________
merge_4 (Merge)                 (None, 32, 32, 40)   0           conv2d_9[0][0]
                                                                 merge_3[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 32, 32, 40)   160         merge_4[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 32, 32, 40)   0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 16, 16, 80)   28880       activation_9[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 16, 16, 80)   320         conv2d_10[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 16, 16, 80)   0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 16, 16, 80)   0           activation_10[0][0]
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 16, 16, 40)   0           merge_4[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 16, 16, 80)   57680       dropout_5[0][0]
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 16, 16, 80)   0           average_pooling2d_1[0][0]
__________________________________________________________________________________________________
merge_5 (Merge)                 (None, 16, 16, 80)   0           conv2d_11[0][0]
                                                                 lambda_2[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 16, 16, 80)   320         merge_5[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 16, 16, 80)   0           batch_normalization_11[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 16, 16, 80)   57680       activation_11[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 16, 16, 80)   320         conv2d_12[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 16, 16, 80)   0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 16, 16, 80)   0           activation_12[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 16, 16, 80)   57680       dropout_6[0][0]
__________________________________________________________________________________________________
merge_6 (Merge)                 (None, 16, 16, 80)   0           conv2d_13[0][0]
                                                                 merge_5[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 16, 16, 80)   320         merge_6[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 16, 16, 80)   0           batch_normalization_13[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 16, 16, 80)   57680       activation_13[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 16, 16, 80)   320         conv2d_14[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 16, 16, 80)   0           batch_normalization_14[0][0]
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 16, 16, 80)   0           activation_14[0][0]
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 16, 16, 80)   57680       dropout_7[0][0]
__________________________________________________________________________________________________
merge_7 (Merge)                 (None, 16, 16, 80)   0           conv2d_15[0][0]
                                                                 merge_6[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 16, 16, 80)   320         merge_7[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 16, 16, 80)   0           batch_normalization_15[0][0]
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 16, 16, 80)   57680       activation_15[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 16, 16, 80)   320         conv2d_16[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 16, 16, 80)   0           batch_normalization_16[0][0]
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 16, 16, 80)   0           activation_16[0][0]
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 16, 16, 80)   57680       dropout_8[0][0]
__________________________________________________________________________________________________
merge_8 (Merge)                 (None, 16, 16, 80)   0           conv2d_17[0][0]
                                                                 merge_7[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 16, 16, 80)   320         merge_8[0][0]
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 16, 16, 80)   0           batch_normalization_17[0][0]
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 8, 8, 160)    115360      activation_17[0][0]
__________________________________________________________________________________________________
batch_normalization_18 (BatchNo (None, 8, 8, 160)    640         conv2d_18[0][0]
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 8, 8, 160)    0           batch_normalization_18[0][0]
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 8, 8, 160)    0           activation_18[0][0]
__________________________________________________________________________________________________
average_pooling2d_2 (AveragePoo (None, 8, 8, 80)     0           merge_8[0][0]
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 8, 8, 160)    230560      dropout_9[0][0]
__________________________________________________________________________________________________
lambda_3 (Lambda)               (None, 8, 8, 160)    0           average_pooling2d_2[0][0]
__________________________________________________________________________________________________
merge_9 (Merge)                 (None, 8, 8, 160)    0           conv2d_19[0][0]
                                                                 lambda_3[0][0]
__________________________________________________________________________________________________
batch_normalization_19 (BatchNo (None, 8, 8, 160)    640         merge_9[0][0]
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 8, 8, 160)    0           batch_normalization_19[0][0]
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 8, 8, 160)    230560      activation_19[0][0]
__________________________________________________________________________________________________
batch_normalization_20 (BatchNo (None, 8, 8, 160)    640         conv2d_20[0][0]
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 8, 8, 160)    0           batch_normalization_20[0][0]
__________________________________________________________________________________________________
dropout_10 (Dropout)            (None, 8, 8, 160)    0           activation_20[0][0]
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 8, 8, 160)    230560      dropout_10[0][0]
__________________________________________________________________________________________________
merge_10 (Merge)                (None, 8, 8, 160)    0           conv2d_21[0][0]
                                                                 merge_9[0][0]
__________________________________________________________________________________________________
batch_normalization_21 (BatchNo (None, 8, 8, 160)    640         merge_10[0][0]
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 8, 8, 160)    0           batch_normalization_21[0][0]
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 8, 8, 160)    230560      activation_21[0][0]
__________________________________________________________________________________________________
batch_normalization_22 (BatchNo (None, 8, 8, 160)    640         conv2d_22[0][0]
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 8, 8, 160)    0           batch_normalization_22[0][0]
__________________________________________________________________________________________________
dropout_11 (Dropout)            (None, 8, 8, 160)    0           activation_22[0][0]
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 8, 8, 160)    230560      dropout_11[0][0]
__________________________________________________________________________________________________
merge_11 (Merge)                (None, 8, 8, 160)    0           conv2d_23[0][0]
                                                                 merge_10[0][0]
__________________________________________________________________________________________________
batch_normalization_23 (BatchNo (None, 8, 8, 160)    640         merge_11[0][0]
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 8, 8, 160)    0           batch_normalization_23[0][0]
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 8, 8, 160)    230560      activation_23[0][0]
__________________________________________________________________________________________________
batch_normalization_24 (BatchNo (None, 8, 8, 160)    640         conv2d_24[0][0]
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 8, 8, 160)    0           batch_normalization_24[0][0]
__________________________________________________________________________________________________
dropout_12 (Dropout)            (None, 8, 8, 160)    0           activation_24[0][0]
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 8, 8, 160)    230560      dropout_12[0][0]
__________________________________________________________________________________________________
merge_12 (Merge)                (None, 8, 8, 160)    0           conv2d_25[0][0]
                                                                 merge_11[0][0]
__________________________________________________________________________________________________
batch_normalization_25 (BatchNo (None, 8, 8, 160)    640         merge_12[0][0]
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 8, 8, 160)    0           batch_normalization_25[0][0]
__________________________________________________________________________________________________
average_pooling2d_3 (AveragePoo (None, 1, 1, 160)    0           activation_25[0][0]
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 16, 16, 160)  115360      merge_5[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 160)          0           average_pooling2d_3[0][0]
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 32, 32, 160)  0           conv2d_26[0][0]
__________________________________________________________________________________________________
softmax (Dense)                 (None, 10)           1610        flatten_1[0][0]
__________________________________________________________________________________________________
conv_out (Conv2D)               (None, 32, 32, 3)    4323        up_sampling2d_1[0][0]
==================================================================================================
Total params: 2,399,565
Trainable params: 2,395,053
Non-trainable params: 4,512
__________________________________________________________________________________________________
Train on 50000 samples, validate on 10000 samples
Epoch 1/10
50000/50000 [==============================] - 116s 2ms/step - loss: 0.2248 - softmax_loss: 0.1302 - conv_out_loss: 9.4610e-04 - softmax_acc: 0.9541 - conv_out_acc: 0.8302 - val_loss: 0.4793 - val_softmax_loss: 0.4405 - val_conv_out_loss: 3.8844e-04 - val_softmax_acc: 0.8830 - val_conv_out_acc: 0.8735
Epoch 2/10
50000/50000 [==============================] - 107s 2ms/step - loss: 0.1212 - softmax_loss: 0.0735 - conv_out_loss: 4.7690e-04 - softmax_acc: 0.9748 - conv_out_acc: 0.8630 - val_loss: 0.4812 - val_softmax_loss: 0.4453 - val_conv_out_loss: 3.5900e-04 - val_softmax_acc: 0.8879 - val_conv_out_acc: 0.8863
Epoch 3/10
50000/50000 [==============================] - 107s 2ms/step - loss: 0.0876 - softmax_loss: 0.0471 - conv_out_loss: 4.0517e-04 - softmax_acc: 0.9838 - conv_out_acc: 0.8719 - val_loss: 0.4524 - val_softmax_loss: 0.4232 - val_conv_out_loss: 2.9205e-04 - val_softmax_acc: 0.8950 - val_conv_out_acc: 0.8837
Epoch 4/10
50000/50000 [==============================] - 107s 2ms/step - loss: 0.0786 - softmax_loss: 0.0395 - conv_out_loss: 3.9125e-04 - softmax_acc: 0.9870 - conv_out_acc: 0.8725 - val_loss: 0.4558 - val_softmax_loss: 0.4268 - val_conv_out_loss: 2.9087e-04 - val_softmax_acc: 0.8979 - val_conv_out_acc: 0.8917
Epoch 5/10
50000/50000 [==============================] - 107s 2ms/step - loss: 0.0747 - softmax_loss: 0.0375 - conv_out_loss: 3.7240e-04 - softmax_acc: 0.9872 - conv_out_acc: 0.8756 - val_loss: 0.4526 - val_softmax_loss: 0.4261 - val_conv_out_loss: 2.6517e-04 - val_softmax_acc: 0.8982 - val_conv_out_acc: 0.8999
Epoch 6/10
50000/50000 [==============================] - 107s 2ms/step - loss: 0.0725 - softmax_loss: 0.0354 - conv_out_loss: 3.7106e-04 - softmax_acc: 0.9886 - conv_out_acc: 0.8755 - val_loss: 0.4504 - val_softmax_loss: 0.4240 - val_conv_out_loss: 2.6402e-04 - val_softmax_acc: 0.8992 - val_conv_out_acc: 0.9003
Epoch 7/10
50000/50000 [==============================] - 107s 2ms/step - loss: 0.0747 - softmax_loss: 0.0378 - conv_out_loss: 3.6922e-04 - softmax_acc: 0.9871 - conv_out_acc: 0.8756 - val_loss: 0.4492 - val_softmax_loss: 0.4229 - val_conv_out_loss: 2.6358e-04 - val_softmax_acc: 0.8995 - val_conv_out_acc: 0.9002
Epoch 8/10
50000/50000 [==============================] - 107s 2ms/step - loss: 0.0716 - softmax_loss: 0.0348 - conv_out_loss: 3.6838e-04 - softmax_acc: 0.9883 - conv_out_acc: 0.8759 - val_loss: 0.4472 - val_softmax_loss: 0.4209 - val_conv_out_loss: 2.6306e-04 - val_softmax_acc: 0.9000 - val_conv_out_acc: 0.8985
Epoch 9/10
50000/50000 [==============================] - 107s 2ms/step - loss: 0.0719 - softmax_loss: 0.0350 - conv_out_loss: 3.6883e-04 - softmax_acc: 0.9886 - conv_out_acc: 0.8752 - val_loss: 0.4480 - val_softmax_loss: 0.4220 - val_conv_out_loss: 2.6074e-04 - val_softmax_acc: 0.8985 - val_conv_out_acc: 0.8985
Epoch 10/10
50000/50000 [==============================] - 107s 2ms/step - loss: 0.0711 - softmax_loss: 0.0346 - conv_out_loss: 3.6491e-04 - softmax_acc: 0.9885 - conv_out_acc: 0.8753 - val_loss: 0.4485 - val_softmax_loss: 0.4227 - val_conv_out_loss: 2.5839e-04 - val_softmax_acc: 0.9012 - val_conv_out_acc: 0.9024
"""



