"""
cifar10vgg.py
geifmany/cifar-vgg 
https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.utils import to_categorical
from keras.models import Sequential
from keras import initializers, layers
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import layers, models, optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
import os
from keras import regularizers

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

if __name__ == '__main__':
    # CIFAR-10データセットをロード
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    # 画像を描画
    nclasses = 10
    pos = 1
    for targetClass in range(nclasses):
        targetIdx = []
        # クラスclassIDの画像のインデックスリストを取得
        for i in range(len(Y_train)):
            if Y_train[i][0] == targetClass:
                targetIdx.append(i)

        # 各クラスからランダムに選んだ最初の10個の画像を描画
        np.random.shuffle(targetIdx)
        for idx in targetIdx[:10]:
            img = toimage(X_train[idx])
            plt.subplot(10, 10, pos)
            plt.imshow(img)
            plt.axis('off')
            pos += 1

    plt.pause(3)
    plt.close()
    
# 入力画像の次元
img_rows, img_cols = 32, 32

# チャネル数（RGBなので3）
img_channels = 3

# 画素値を0-1に変換
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
nb_classes = 10

# クラスラベル（0-9）をone-hotエンコーディング形式に変換
Y_train = to_categorical(Y_train, nb_classes)
Y_test = to_categorical(Y_test, nb_classes)

def vgg16_model(input_shape, n_class=10):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    weight_decay = 0.0  #0.0005
    axis_num = -1
    x = layers.Input(shape=input_shape)
    y = layers.Input(shape=(n_class,))
    
    
    conv1 = Conv2D(64, (3, 3),activation='relu', padding='same',
                   input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay))(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.3)(conv1)  #0.3
    
    conv2 = Conv2D(64, (3, 3),activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.3)(conv2)  #add 0.3
    
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)  #>>>conv_out1
    
    conv3 = Conv2D(128, (3, 3),activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.4)(conv3)  #0.4
    conv4 = Conv2D(128, (3, 3),activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(conv3)
    conv4 = BatchNormalization()(conv4)
    
    conv4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5= Conv2D(256, (3, 3),activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(0.4)(conv5)  #0.4
    
    conv6= Conv2D(256, (3, 3),activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.4)(conv6)  #0.4
    
    conv7= Conv2D(256, (3, 3),activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(conv6)
    conv7 = BatchNormalization()(conv7)
    
    conv7 = MaxPooling2D(pool_size=(2, 2))(conv7)
    
    conv8= Conv2D(512, (3, 3),activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(0.4)(conv8)
    
    conv9= Conv2D(512, (3, 3),activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(0.4)(conv9)         #>>>conv_out2
    
    conv10= Conv2D(512, (3, 3),activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(conv9)
    conv10 = BatchNormalization()(conv10)
    
    conv10 = MaxPooling2D(pool_size=(2, 2))(conv10)

    conv11= Conv2D(512, (3, 3),activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = Dropout(0.4)(conv11)
    
    conv12= Conv2D(512, (3, 3),activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Dropout(0.4)(conv12)

    conv13= Conv2D(512, (3, 3),activation='relu', padding='same',kernel_regularizer=regularizers.l2(weight_decay))(conv12)
    conv13 = BatchNormalization()(conv13)
    
    conv13 = MaxPooling2D(pool_size=(2, 2))(conv13)
    conv13 = Dropout(0.4)(conv13)
    
    x1=Flatten()(conv13)
    x1=Dense(512,activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.5)(x1) 
    
    softmax=Dense(n_class, activation='softmax',name='softmax')(x1)
    
    conv14=Conv2D(128, (3, 3),activation='relu', padding="same")(conv9)
    conv14=UpSampling2D(size=(2, 2))(conv14)
    
    conv14=Conv2D(128, (3, 3),activation='relu', padding="same")(conv14)
    conv14=UpSampling2D(size=(2, 2))(conv14)
    conv14=Conv2D(64, (3, 3),activation='relu', padding="same")(conv14)
    
    conv14=UpSampling2D(size=(2, 2))(conv14)
    """
    conv14=Conv2D(32, (3, 3),activation='relu', padding="same")(conv14)
    
    conv14=UpSampling2D(size=(2, 2))(conv14)
    
    conv14=Conv2D(32, (3, 3),activation='relu', padding="same")(conv14)
    conv14=UpSampling2D(size=(2, 2))(conv14)
    """
    conv15=UpSampling2D(size=(2, 2))(conv2)
    conv_out=Conv2D(3, (3, 3), padding="same",activation='sigmoid', name="conv_out")(conv15)
    
    #conv16=UpSampling2D(size=(2, 2))(conv14)
    #conv_out=Conv2D(3, (3, 3), padding="same",activation='sigmoid', name="conv_out")(conv16)
    
    return models.Model([x, y], [softmax, conv_out])




def train(model, data, epoch_size=32):
    (x_train, y_train), (x_test, y_test) = data
    #loss='binary_crossentropy'  'categorical_crossentropy' 'mse'
    model.compile(optimizer='adam',
                  loss={'softmax':'categorical_crossentropy', 'conv_out':'mse'},
                  loss_weights=[1., 100.],
                  metrics={'softmax': 'accuracy','conv_out': 'accuracy'})

    history=model.fit([x_train, y_train],[y_train, x_train], batch_size=32, epochs=epoch_size,
              validation_data=[[x_test, y_test], [y_test, x_test]])

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



input_img = Input(shape=(32,32,3)) 

model = vgg16_model(input_shape=[32, 32, 3], n_class=10)

# モデルのサマリを表示
model.summary()


X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# 訓練
#model, history = train(model=model, data=((X_train, Y_train), (X_test, Y_test)), epoch_size=1)

for j in range(20):
    model, history = train(model=model, data=((X_train, Y_train), (X_test, Y_test)), epoch_size=1)
    model.save_weights('params_vggAE_epoch_{0:03d}.hdf5'.format(j), True)
    plot_generated_batch(j,model=model, data1=(X_train, Y_train),data2=(X_test, Y_test))
    # 学習履歴を保存
    save_history(history, os.path.join("./caps_figures/", 'history_vgg_ACE.txt'),j)
    
"""
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            (None, 32, 32, 3)    0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 64)   1792        input_2[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32, 32, 64)   256         conv2d_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 32, 32, 64)   0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 64)   36928       dropout_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 32, 64)   256         conv2d_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 32, 32, 64)   0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 16, 16, 64)   0           dropout_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 16, 16, 128)  73856       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 16, 16, 128)  512         conv2d_3[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 16, 16, 128)  0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 16, 16, 128)  147584      dropout_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 16, 16, 128)  512         conv2d_4[0][0]
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 8, 8, 128)    0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 8, 8, 256)    295168      max_pooling2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 8, 8, 256)    1024        conv2d_5[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 8, 8, 256)    0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 8, 8, 256)    590080      dropout_4[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 8, 8, 256)    1024        conv2d_6[0][0]
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 8, 8, 256)    0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 8, 8, 256)    590080      dropout_5[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 8, 8, 256)    1024        conv2d_7[0][0]
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 4, 4, 256)    0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 4, 4, 512)    1180160     max_pooling2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 4, 4, 512)    2048        conv2d_8[0][0]
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 4, 4, 512)    0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 4, 4, 512)    2359808     dropout_6[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 4, 4, 512)    2048        conv2d_9[0][0]
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 4, 4, 512)    0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 4, 4, 512)    2359808     dropout_7[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 4, 4, 512)    2048        conv2d_10[0][0]
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 2, 2, 512)    0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 2, 2, 512)    2359808     max_pooling2d_4[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 2, 2, 512)    2048        conv2d_11[0][0]
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 2, 2, 512)    0           batch_normalization_11[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 2, 2, 512)    2359808     dropout_8[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 2, 2, 512)    2048        conv2d_12[0][0]
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, 2, 2, 512)    0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 2, 2, 512)    2359808     dropout_9[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 2, 2, 512)    2048        conv2d_13[0][0]
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 1, 1, 512)    0           batch_normalization_13[0][0]
__________________________________________________________________________________________________
dropout_10 (Dropout)            (None, 1, 1, 512)    0           max_pooling2d_5[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 512)          0           dropout_10[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 512)          262656      flatten_1[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 512)          2048        dense_1[0][0]
__________________________________________________________________________________________________
dropout_11 (Dropout)            (None, 512)          0           batch_normalization_14[0][0]
__________________________________________________________________________________________________
up_sampling2d_4 (UpSampling2D)  (None, 32, 32, 64)   0           max_pooling2d_1[0][0]
__________________________________________________________________________________________________
softmax (Dense)                 (None, 10)           5130        dropout_11[0][0]
__________________________________________________________________________________________________
conv_out (Conv2D)               (None, 32, 32, 3)    1731        up_sampling2d_4[0][0]
==================================================================================================
Total params: 15,003,149
Trainable params: 14,993,677
Non-trainable params: 9,472
__________________________________________________________________________________________________
Train on 50000 samples, validate on 10000 samples
j=0 Epoch 1/1
2018-04-02 06:05:24.458929: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
50000/50000 [==============================] - 117s 2ms/step - loss: 2.6721 - softmax_loss: 1.9839 - conv_out_loss: 0.0069 - softmax_acc: 0.3062 - conv_out_acc: 0.6681 - val_loss: 1.9908 - val_softmax_loss: 1.5698 - val_conv_out_loss: 0.0042 - val_softmax_acc: 0.4342 - val_conv_out_acc: 0.7466

j=19 epoch 1/1
50000/50000 [==============================] - 120s 2ms/step - loss: 0.4092 - softmax_loss: 0.2581 - conv_out_loss: 0.0015 - softmax_acc: 0.9147 - conv_out_acc: 0.8163 - val_loss: 0.7814 - val_softmax_loss: 0.4931 - val_conv_out_loss: 0.0029 - val_softmax_acc: 0.8553 - val_conv_out_acc: 0.8482
"""