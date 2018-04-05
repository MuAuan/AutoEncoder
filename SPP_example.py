import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
from keras.models import Sequential

from SpatialPyramidPooling import SpatialPyramidPooling
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras import initializers, layers
from keras.optimizers import SGD, Adam
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.engine import Input, Model
from keras.layers import Input, MaxPooling2D, UpSampling2D
import os
from keras.layers import Dense, Activation, Flatten, Lambda, Convolution2D, AveragePooling2D, BatchNormalization, Dropout

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
    if epoch < 4:
        return 0.0001  #0.1
    elif epoch < 8:  #120
        return 0.00005 #0.02
    elif epoch < 12:  #160
        return 0.00001  #0.004
    else:
        return 0.000005  #0.0008

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
    plt.savefig("./SPP_ACE{0:03d}.png".format(i))
    plt.pause(3)
    plt.close()


batch_size = 32
num_channels = 3
num_classes = 10

#model = Sequential()

input_shape=(None, None, num_channels)
#inputs = Input(shape=(img_rows, img_cols, img_channels))
input1 = layers.Input(shape=input_shape)
input2 = layers.Input(shape=(num_classes,))


# uses theano ordering. Note that we leave the image size as None to allow multiple image sizes
x = Convolution2D(32, 3, 3, border_mode='same')(input1)
x = Activation('relu')(x)
x = Convolution2D(32, 3, 3, border_mode='same')(x)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Convolution2D(32, 3, 3, border_mode='same')(x)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)

x1= MaxPooling2D(pool_size=(2, 2))(x)

x = Convolution2D(64, 3, 3, border_mode='same')(x1)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Convolution2D(64, 3, 3, border_mode='same')(x)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Convolution2D(64, 3, 3, border_mode='same')(x)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Convolution2D(128, 3, 3, border_mode='same')(x)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Convolution2D(128, 3, 3, border_mode='same')(x)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Convolution2D(128, 3, 3, border_mode='same')(x)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Convolution2D(512, 3, 3, border_mode='same')(x)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Convolution2D(512, 3, 3, border_mode='same')(x)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Convolution2D(512, 3, 3,border_mode='same')(x)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = SpatialPyramidPooling([1, 2, 4])(x)

x = Dense(num_classes)(x)
softmax=Activation('softmax',name="softmax")(x)

conv_out=UpSampling2D(size=(2, 2))(x1)
conv_out=Convolution2D(3, (3, 3), padding="same",activation='sigmoid', name="conv_out")(conv_out)
#conv_out=Convolution2D(3, 3, 3,name="conv_out")(conv_out)

#model.compile(loss='categorical_crossentropy', optimizer='sgd')

model = Model([input1,input2], [softmax, conv_out])

model.summary()

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

adm = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)


for j in range(20):
        model, history = train(adm,model=model, data=((x_train, y_train), (x_test, y_test)), epoch_size=10 ,batch_size=batch_size)
        model.save_weights('params_SPPAE_epoch_{0:03d}.hdf5'.format(j), True)
        plot_generated_batch(j,model=model, data1=(x_train, y_train),data2=(x_test, y_test))
        # 学習履歴を保存
        save_history(history, os.path.join("./", 'history_SPP_AE.txt'),j)

#print("train on 64x64x3 images")
#model.fit(np.random.rand(batch_size, 64, 64,num_channels), np.zeros((batch_size, num_classes)),epochs=10)
#print("train on 32x32x3 images")
#model.fit(np.random.rand(batch_size, 32, 32,num_channels), np.zeros((batch_size, num_classes)),epochs=10)
"""
__________________________________________________________________________________________________
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, None, None, 3 0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, None, None, 3 896         input_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, None, None, 3 0           conv2d_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, None, None, 3 9248        activation_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, None, None, 3 128         conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, None, None, 3 0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, None, None, 3 0           activation_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, None, None, 3 9248        dropout_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, None, None, 3 128         conv2d_3[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, None, None, 3 0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, None, None, 3 0           activation_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, None, None, 1 36992       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, None, None, 1 512         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, None, None, 1 0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, None, None, 1 0           activation_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, None, None, 1 147584      dropout_2[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, None, None, 1 512         conv2d_5[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, None, None, 1 0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, None, None, 1 0           activation_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, None, None, 1 147584      dropout_3[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, None, None, 1 512         conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, None, None, 1 0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, None, None, 1 0           activation_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, None, None, 5 590336      max_pooling2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, None, None, 5 2048        conv2d_7[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, None, None, 5 0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, None, None, 5 0           activation_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, None, None, 5 2359808     dropout_4[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, None, None, 5 2048        conv2d_8[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, None, None, 5 0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, None, None, 5 0           activation_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, None, None, 5 2359808     dropout_5[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, None, None, 5 2048        conv2d_9[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, None, None, 5 0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
spatial_pyramid_pooling_1 (Spat (None, 10752)        0           activation_9[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 10)           107530      spatial_pyramid_pooling_1[0][0]
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, None, None, 3 0           max_pooling2d_1[0][0]
__________________________________________________________________________________________________
softmax (Activation)            (None, 10)           0           dense_1[0][0]
__________________________________________________________________________________________________
conv_out (Conv2D)               (None, None, None, 3 867         up_sampling2d_1[0][0]
==================================================================================================
Total params: 5,777,837
Trainable params: 5,773,869
Non-trainable params: 3,968
__________________________________________________________________________________________________


pooling_regions = [1, 2, 4]
num_rois = 2
num_channels = 3

if dim_ordering == 'tf':
    in_img = Input(shape=(None, None, num_channels))
elif dim_ordering == 'th':
    in_img = Input(shape=(num_channels, None, None))

in_roi = Input(shape=(num_rois, 4))

out_roi_pool = RoiPooling(pooling_regions, num_rois)([in_img, in_roi])

model = Model([in_img, in_roi], out_roi_pool)

if dim_ordering == 'th':
    X_img = np.random.rand(1, num_channels, img_size, img_size)
    row_length = [float(X_img.shape[2]) / i for i in pooling_regions]
    col_length = [float(X_img.shape[3]) / i for i in pooling_regions]
elif dim_ordering == 'tf':
    X_img = np.random.rand(1, img_size, img_size, num_channels)
    row_length = [float(X_img.shape[1]) / i for i in pooling_regions]
    col_length = [float(X_img.shape[2]) / i for i in pooling_regions]

X_roi = np.array([[0, 0, img_size / 1, img_size / 1],
                  [0, 0, img_size / 2, img_size / 2]])

X_roi = np.reshape(X_roi, (1, num_rois, 4))

Y = model.predict([X_img, X_roi])
"""

