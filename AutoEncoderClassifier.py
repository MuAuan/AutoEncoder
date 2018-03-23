from keras.layers import Dense
from keras.models import Model
from keras.datasets import mnist, cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras import initializers, layers
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D

def prepare_model(input_shape=(28, 28, 1), class_num=10):
    input = Input(input_shape)
    kernel_size = (3, 3)
    max_pool_size = (2, 2)
    upsampling_size = (2, 2)

    enc_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(input)
    enc_cnn = Dropout(0.1)(enc_cnn)
    enc_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(enc_cnn)
    enc_cnn = Dropout(0.1)(enc_cnn)
    enc_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(enc_cnn)
    enc_cnn = Dropout(0.1)(enc_cnn)
    enc_cnn = MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(enc_cnn)

    enc_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(enc_cnn)
    enc_cnn = Dropout(0.1)(enc_cnn)
    enc_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(enc_cnn)
    enc_cnn = Dropout(0.1)(enc_cnn)
    enc_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(enc_cnn)
    enc_cnn = MaxPooling2D(pool_size=max_pool_size, strides=(2, 2))(enc_cnn)

    fc = Flatten()(enc_cnn)
    fc = Dense(1024, activation='relu')(fc)
    softmax = Dense(class_num, activation='softmax', name='classification')(fc)

    dec_cnn = UpSampling2D(upsampling_size)(enc_cnn)
    dec_cnn = Conv2D(64, kernel_size, padding='same', activation='relu')(dec_cnn)
    dec_cnn = UpSampling2D(upsampling_size)(dec_cnn)
    dec_cnn = Conv2D(1, kernel_size, padding='same', activation='sigmoid', name='autoencoder')(dec_cnn)

    outputs = [softmax, dec_cnn]

    model = Model(input=input, output=outputs)
    return model


"""
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
"""

model.compile(loss={'classification': categorical_crossentropy, 'autoencoder': mean_squared_error},
                  loss_weights={'classification': 0.9, 'autoencoder': 0.1},
                  optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
model.summary()

"""
dim_factor=6
encoding_dim = 32*dim_factor
input_img = Input(shape=(32,32,3))
#input_img = Input(shape=(3072,))
encoded = encoded(input_img, dim_factor)
decoded = decoded(encoded, dim_factor)
autoencoder = Model(input=input_img, output=decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#autoencoder.load_weights('autoencoder.h5')
autoencoder.summary()
"""
(x_train,y_train), (x_test,y_test) = mnist.load_data()  #cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_train = x_train[:10000]
x_test  = x_test[:10000]
y_train = y_train[:10000]
y_test  = y_test[:10000]

for j in range(10):
    model.fit(x_train, y_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, y_test))

    model.save_weights('encoderClassifier_'+str(dim_factor)+'{0:03d}.h5'.format(j))
    model.load_weights('encoderClassifier_'+str(dim_factor)+'{0:03d}.h5'.format(j))
    #autoencoder.save_weights('autoencoder_D'+str(dim_factor)+'{0:03d}.h5'.format(j))
    #autoencoder.load_weights('autoencoder_D'+str(dim_factor)+'{0:03d}.h5'.format(j))

# テスト画像を変換
    decoded_imgs = model.predict(x_test,y_test)

    n = 10
#encoded_imgs=[]
    """
    encoder = Model(input_img, encoded)
    encoded_imgs = encoder.predict(x_test[:n])
    """
    plt.figure(figsize=(32, 12))
    for i in range(n):
    # オリジナルのテスト画像を表示
        ax = plt.subplot(3, n, i+1)
        plt.imshow(x_test[i].reshape(28,28,1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(3, n, i+1+2*n)
        plt.imshow(decoded_imgs[i].reshape(28, 28,1))
        #ax = plt.subplot(3, n, i+1+n)
        #plt.imshow(encoded_imgs[i].reshape(32,32,3))
    plt.axis('off')
    plt.savefig("./caps_figures/autoencoder"+str(dim_factor)+"{0:03d}.png".format(j))
        

    plt.pause(3)
    plt.close()

"""
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 32, 32, 3)    0
__________________________________________________________________________________________________
conv1 (Conv2D)                  (None, 24, 24, 256)  62464       input_1[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 8, 8, 256)    5308672     conv1[0][0]
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 2048, 8)      0           conv2d_1[0][0]
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 2048, 8)      0           reshape_1[0][0]
__________________________________________________________________________________________________
digitcaps (CapsuleLayer)        (None, 10, 16)       2641920     lambda_1[0][0]
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 10)           0
__________________________________________________________________________________________________
mask_1 (Mask)                   (None, 16)           0           digitcaps[0][0]
                                                                 input_2[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 512)          8704        mask_1[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1024)         525312      dense_1[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 3072)         3148800     dense_2[0][0]
__________________________________________________________________________________________________
out_caps (Length)               (None, 10)           0           digitcaps[0][0]
__________________________________________________________________________________________________
out_recon (Reshape)             (None, 32, 32, 3)    0           dense_3[0][0]
==================================================================================================
Total params: 11,695,872
Trainable params: 11,675,392
Non-trainable params: 20,480
__________________________________________________________________________________________________
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