import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
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

# CNNを構築
def model_cifar(input_shape, n_class=10):
    axis_num = -1
    x = layers.Input(shape=input_shape)
    y = layers.Input(shape=(n_class,))
    conv1=Conv2D(512, (3, 3), input_shape=(32, 32, 3), padding="same")(x)
    conv1 = BatchNormalization(axis=axis_num)(conv1)   #add
    #conv1 = Dropout(0.5)(conv1) #add
    conv1=MaxPooling2D(pool_size=(2, 2), padding="same")(conv1)

    conv2=Conv2D(256, (3, 3),activation='relu', padding="same")(conv1)
    conv1 = BatchNormalization(axis=axis_num)(conv1)   #add
    #conv1 = Dropout(0.5)(conv1) #add
    conv2=MaxPooling2D(pool_size=(2, 2), padding="same")(conv2)
    conv2=Conv2D(128, (3, 3),activation='relu', padding="same")(conv2)
    conv2 = BatchNormalization(axis=axis_num)(conv2)   #add
    #conv2 = Dropout(0.5)(conv2) #add
    conv2=Conv2D(64, (3, 3),activation='relu', padding="same")(conv2)

    x1=Flatten()(conv2)
    x1=Dense(1024)(x1)
    #x1 = Dropout(0.5)(x1) #add
    x1=Activation('relu')(x1)
    x1=Dense(256)(x1)
    #x1 = Dropout(0.5)(x1) #add
    x1=Activation('relu')(x1)

    softmax=Dense(nb_classes, activation='softmax',name='softmax')(x1)
    
    conv3=Conv2D(64, (3, 3),activation='relu', padding="same")(conv2)
    conv3=UpSampling2D(size=(2, 2))(conv3)
    conv3=Conv2D(32, (3, 3),activation='relu', padding="same")(conv3)
    conv3=UpSampling2D(size=(2, 2))(conv3)
    
    conv_out=Conv2D(3, (3, 3), padding="same",activation='sigmoid', name="conv_out")(conv3)

    return models.Model([x, y], [softmax, conv_out])

def train(model, data, epoch_size=32):

    (x_train, y_train), (x_test, y_test) = data
    #loss='binary_crossentropy'  'categorical_crossentropy' 'mse'
    model.compile(optimizer="adam",
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
    plt.savefig("./caps_figures/AutoCrassiEncoder{0:03d}.png".format(i))
    plt.pause(3)
    plt.close()

def plot_generated_batch_encoder(n, model1,model2,data1,data2):
    x_test, y_test = data2
    encoded_imgs1 = encoder1.predict([x_test[:n], y_test[:n]], batch_size=32)
    encoded_imgs2 = encoder2.predict([x_test[:n], y_test[:n]], batch_size=32)
    
    plt.figure(figsize=(10, 8))
    #mx1=np.argmax(abs(encoded_imgs1[0].reshape( 32, 10, 3)))
    #mx2=np.argmax(abs(encoded_imgs2[0].reshape( 32, 10, 3)))
    for i in range(n):
    # オリジナルのテスト画像を表示
        ax = plt.subplot(4, n, i+1)
        plt.imshow(x_test[i].reshape(32,32,3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(4, n, i+1+3*n)
        #print(decoded_imgs[i])
        plt.imshow((decoded_imgs[i].reshape(32, 32,3)))
        ax = plt.subplot(4, n, i+1+n)
        #print(encoded_imgs1[i])
        plt.imshow(abs(encoded_imgs1[i].reshape( 24, 24, 3)))
        ax = plt.subplot(4, n, i+1+2*n)
        plt.imshow(abs(encoded_imgs2[i].reshape( 16, 16, 3))*100)
    plt.axis('off')
    plt.savefig("./caps_figures/intermidiate_encoder"+str(dim_factor)+"{0:03d}.png".format(j))
        

    plt.pause(3)
    plt.close()   


#model.add(UpSampling2D((2,2)))(conv2d_4)
input_img = Input(shape=(32,32,3)) 

model = model_cifar(input_shape=[32, 32, 3], n_class=10)
#model = Capsdecoder(input_shape=[32, 32, 3], n_class=10, num_routing=3)

# モデルのサマリを表示
model.summary()
#plot(model, show_shapes=True, to_file=os.path.join(result_dir, 'model.png'))

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# 訓練
#model, history = train(model=model, data=((X_train, Y_train), (X_test, Y_test)), epoch_size=1)

for j in range(20):
    model, history = train(model=model, data=((X_train, Y_train), (X_test, Y_test)), epoch_size=1)
    model.save_weights('params_capsAE_epoch_{0:03d}.hdf5'.format(j), True)
    plot_generated_batch(j,model=model, data1=(X_train, Y_train),data2=(X_test, Y_test))
    # 学習履歴を保存
    save_history(history, os.path.join("./caps_figures/", 'history_ACE.txt'),j)

"""
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            (None, 32, 32, 3)    0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 32)   896         input_2[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 32, 32)   0           conv2d_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 32)   9248        activation_1[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 16, 16, 32)   0           conv2d_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 16, 16, 64)   18496       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 16, 16, 64)   0           conv2d_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 16, 16, 32)   18464       activation_2[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 8192)         0           conv2d_4[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 2048)         16779264    flatten_1[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 2048)         0           dense_1[0][0]
__________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)  (None, 32, 32, 32)   0           conv2d_4[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 10)           20490       activation_3[0][0]
__________________________________________________________________________________________________
conv_out (Conv2D)               (None, 32, 32, 3)    867         up_sampling2d_1[0][0]
==================================================================================================
Total params: 16,847,725
Trainable params: 16,847,725
Non-trainable params: 0
__________________________________________________________________________________________________
"""
