from keras.layers import Dense
from keras.models import Model
from keras.datasets import mnist, cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras import initializers, layers
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D



def encoded(input_img,dim_factor):
    #input_shape=[28, 28, 1]
    #x = Dense(3072, activation='relu')(input_img)
    #x = Dense(1024, activation='relu')(input_img)
    #x = Dense(512, activation='relu')(x)
    #encoded = Dense(encoding_dim, activation='relu')(x)
    
    x = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='same', activation='relu', name='conv1')(input_img)
    
    x = Convolution2D(64, (3, 3), activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(32, (3, 3), activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    
    encoded = Convolution2D(8*dim_factor, (3, 3), activation='relu', border_mode='same')(x)
    
    #encoded = MaxPooling2D((3, 3), border_mode='valid')(x)
    
    return encoded

def decoded(encoded,dim_factor):
    #x = Dense(512, activation='relu')(encoded)
    #x = Dense(1024, activation='relu')(x)
    #decoded = Dense(3072, activation='sigmoid')(x)
    
    x = Convolution2D(8*dim_factor, (3, 3), activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(3, (3, 3), activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    #x = Convolution2D(3, 3, 3, activation='relu')(x)
    #x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)
    
    return decoded

dim_factor=6*2*2
encoding_dim = 32*dim_factor
input_img = Input(shape=(32,32,3))
#input_img = Input(shape=(3072,))
encoded = encoded(input_img, dim_factor)
decoded = decoded(encoded, dim_factor)
autoencoder = Model(input=input_img, output=decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#autoencoder.load_weights('autoencoder.h5')
autoencoder.summary()

(x_train,y_train), (x_test,y_test) = cifar10.load_data() #mnist.load_data()  #cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_train = x_train[:50000]
x_test  = x_test[:10000]

for j in range(10):
    autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

    autoencoder.save_weights('autoencoder_D'+str(dim_factor)+'{0:03d}.h5'.format(j))
    autoencoder.load_weights('autoencoder_D'+str(dim_factor)+'{0:03d}.h5'.format(j))

# テスト画像を変換
    decoded_imgs = autoencoder.predict(x_test)

    n = 10
#encoded_imgs=[]
    encoder = Model(input_img, encoded)
    encoded_imgs = encoder.predict(x_test[:n])

    plt.figure(figsize=(32, 12))
    for i in range(n):
    # オリジナルのテスト画像を表示
        ax = plt.subplot(3, n, i+1)
        plt.imshow(x_test[i].reshape(32,32,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(3, n, i+1+2*n)
        plt.imshow(decoded_imgs[i].reshape(32, 32,3))
        ax = plt.subplot(3, n, i+1+n)
        plt.imshow(encoded_imgs[i].reshape(32*2,32*2,3))
    plt.axis('off')
    plt.savefig("./caps_figures/autoencoder"+str(dim_factor)+"{0:03d}.png".format(j))
        

    plt.pause(3)
    plt.close()

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 3072)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              3146752
_________________________________________________________________
dense_2 (Dense)              (None, 512)               524800
_________________________________________________________________
dense_3 (Dense)              (None, 320)               164160
_________________________________________________________________
dense_4 (Dense)              (None, 512)               164352
_________________________________________________________________
dense_5 (Dense)              (None, 1024)              525312
_________________________________________________________________
dense_6 (Dense)              (None, 3072)              3148800
=================================================================
Total params: 7,674,176
Trainable params: 7,674,176
Non-trainable params: 0
_________________________________________________________________
"""