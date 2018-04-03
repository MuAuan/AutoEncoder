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
from keras.optimizers import SGD
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


def train(sgd, model, data, epoch_size=32,batch_size=128):
    (x_train, y_train), (x_test, y_test) = data
    #loss='binary_crossentropy'  'categorical_crossentropy' 'mse'
    model.compile(optimizer=sgd,
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
    plt.savefig("./caps_figures/vggAutoCrassiEncoder{0:03d}.png".format(i))
    plt.pause(3)
    plt.close()

nb_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# reorder dimensions for tensorflow
x_train = np.transpose(x_train.astype('float32'), (0, 1, 2,3))
mean = np.mean(x_train, axis=0, keepdims=True)
std = np.std(x_train)
x_train = (x_train - mean) / std
x_test = np.transpose(x_test.astype('float32'), (0, 1,2, 3))
x_test = (x_test - mean) / std
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

for i in range(0, blocks_per_group):
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
x1 = Activation('relu')(x)
x = AveragePooling2D(pool_size=(8, 8), strides=None, border_mode='valid', dim_ordering='tf')(x1)
x = Flatten()(x)

softmax = Dense(nb_classes, activation='softmax', name='softmax')(x)

conv_out=Convolution2D(32 * widening_factor, (3, 3),activation='relu', padding="same")(x1)
conv_out=UpSampling2D(size=(2, 2))(conv_out)
conv_out=Convolution2D(32 * widening_factor, (3, 3),activation='relu', padding="same")(conv_out)
conv_out=UpSampling2D(size=(2, 2))(conv_out)

conv_out=Convolution2D(3, (3, 3), padding="same",activation='sigmoid', name="conv_out")(conv_out)


model = Model([input1,input2], [softmax, conv_out])

model.summary()

#%%time


sgd = SGD(lr=0.1, decay=5e-4, momentum=0.9, nesterov=True)

"""
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
"""

batch_size = 128
nb_epoch = 200
data_augmentation = True

# Learning rate schedule
def lr_sch(epoch):
    if epoch < 60:
        return 0.1
    elif epoch < 120:
        return 0.02
    elif epoch < 160:
        return 0.004
    else:
        return 0.0008

# Learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_sch)


# Model saving callback
#checkpointer = ModelCheckpoint(filepath='stochastic_depth_cifar10.hdf5', verbose=1, save_best_only=True)

if not data_augmentation:
    print('Not using data augmentation.')
    
    """
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                        validation_data=(x_test, y_test), shuffle=True,
                        callbacks=[lr_scheduler])
    """
    #model.load_weights('params_vggAE_epoch_011.hdf5')
    for j in range(20):
        model, history = train(sgd,model=model, data=((x_train, y_train), (x_test, y_test)), epoch_size=1,batch_size=batch_size)
        model.save_weights('params_vggAE_epoch_{0:03d}.hdf5'.format(j), True)
        plot_generated_batch(j,model=model, data1=(x_train, y_train),data2=(x_test, y_test))
        # 学習履歴を保存
        save_history(history, os.path.join("./caps_figures/", 'history_vgg_ACE.txt'),j)
        
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
        model, history = train(sgd,model=model, data=((x_train, y_train), (x_test, y_test)), epoch_size=1)
        model.save_weights('params_vggAE_epoch_{0:03d}.hdf5'.format(j), True)
        plot_generated_batch(j,model=model, data1=(x_train, y_train),data2=(x_test, y_test))
        # 学習履歴を保存
        save_history(history, os.path.join("./caps_figures/", 'history_vgg_ACE.txt'),j)
        
    """
    # fit the model on the batches generated by datagen.flow()
    history = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
                                  samples_per_epoch=x_train.shape[0], 
                                  nb_epoch=nb_epoch, verbose=1,
                                  validation_data=(x_test, y_test),
                                  callbacks=[lr_scheduler])
    """



