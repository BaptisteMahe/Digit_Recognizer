from __future__ import print_function
from __future__ import absolute_import
import numpy as np

np.random.seed(1337)  # for reproducibility

import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pylab as pl

from time import time
from random import randint
from dataset import get_data
from utilities import nice_imshow, make_mosaic


class CNNetwork:

    def __init__(self, WEIGHTS_FNAME, nb_classes=10, nb_filters=32, pool_size=(2, 2), kernel_size=(3, 3)):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.input_shape = None

        self.nb_classes = nb_classes
        self.nb_filters = nb_filters
        self.pool_size = pool_size
        self.kernel_size = kernel_size

        self.model = None

        self.batch_size = 128
        self.WEIGHTS_FNAME = WEIGHTS_FNAME
        self.is_loaded = False

        self.prediction = None
        self.training_time = None

        self.convout_f = None

    def get_DB(self, img_rows=28, img_cols=28):
        self.X_train, self.Y_train, self.X_test, self.Y_test, self.input_shape = get_data(img_rows, img_cols,
                                                                                          self.nb_classes)
        print(self.X_train)
        print(self.Y_train)

    def init_model(self):
        self.model = Sequential()
        self.model.add(Convolution2D(self.nb_filters, self.kernel_size, name="Conv1",
                                     border_mode='valid', input_shape=self.input_shape))
        self.model.add(Activation('relu', name="Conv1Relu"))
        self.model.add(MaxPooling2D(pool_size=self.pool_size, name="Pool1"))
        self.model.add(Convolution2D(self.nb_filters, self.kernel_size, name="Conv2"))
        self.model.add(Activation('relu', name="Conv2Relu"))
        self.model.add(MaxPooling2D(pool_size=self.pool_size, name="Pool2"))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(100, activation="relu", name="Dense1"))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))  # éviter de "sur-apprendre" la Bapp
        self.model.add(Dense(self.nb_classes, activation="softmax", name="Dense2"))
        self.model.add(Activation('softmax'))

        self.convout_f = [K.function([self.model.layers[0].input], [self.model.layers[1].output]),
                          K.function([self.model.layers[0].input], [self.model.layers[2].output])]

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta',
                           metrics=['acc'])

    def summary(self):
        if self.model is not None:
            print(self.model.summary())
        else:
            print("Model not initialized")

    def load_weights(self):
        if os.path.exists(self.WEIGHTS_FNAME):
            print('Loading existing weights...')
            self.model.load_weights(self.WEIGHTS_FNAME)
            self.is_loaded = True

    def train_model(self, nb_epoch=1, save_to=""):
        t = time()
        self.model.fit(self.X_train, self.Y_train, nb_epoch=nb_epoch)
        if self.is_loaded:
            if save_to:
                self.WEIGHTS_FNAME = save_to
            self.model.save_weights(self.WEIGHTS_FNAME)
        else:
            print("Warning : you didn't save this fit because of no weights loaded")
        self.training_time = time() - t

    def print_training_time(self):
        print("\nTraining time = " + str(int(self.training_time * 1000) / 1000) + "s")

    def get_score(self):
        score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        print('Test score : ', score[0])
        print('Test accuracy : ', score[1])

    def print_img(self, img_id, n=1):
        x_train, y_train = self.X_train, self.Y_train
        x_test, y_test = self.X_test, self.Y_test

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        y_train = np_utils.to_categorical(y_train, self.nb_classes)
        y_test = np_utils.to_categorical(y_test, self.nb_classes)

        # print(x_train.shape)
        # print(x_test.shape)
        # print(y_train.shape)
        # print(y_test.shape)

        plt.figure(figsize=(20, 4))

        for i in range(n):
            # display original

            img_id = img_id + i
            img_class = np.where(self.Y_test[img_id] == 1)[0][0]
            print("True Class : ", img_class)
            img_prediction = self.predict(img_id)[0]
            print("Predicted Class : ", int(img_prediction.argmax(axis=-1)))
            print("All score Classes : ", img_prediction)

            ax = plt.subplot(1, n, i + 1)
            plt.imshow(x_test[img_id].reshape(28, 28))
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def predict(self, img_id):
        return self.model.predict([[self.X_test[img_id]]])

    def predict_all(self):
        self.prediction = self.model.predict_classes(self.X_test)

    def visu_first_layer(self, img_id=randint(0, 1000)):

        # Visualize the first layer of convolutions on an input image
        X = self.X_test[img_id:img_id + 1, :, :, :]
        print(X.shape)
        pl.figure()
        pl.title('input')
        nice_imshow(pl.gca(), np.squeeze(X), vmin=0, vmax=1, cmap=cm.binary)
        pl.show()

    def visu_weights(self):

        # Visualize weight
        W = self.model.get_weights()[0]
        # W = model.layers[0].W.get_value(borrow=True)
        W = np.squeeze(W)
        print("W shape : ", W.shape)

        pl.figure(figsize=(15, 15))
        pl.title('conv1 weights')
        nice_imshow(pl.gca(), make_mosaic(W, 6, 6), cmap=cm.binary)
        pl.show()

    def visu_convo(self, n, img_id=randint(0,10000)):

        # Visualize convolution result (after activation)
        X = self.X_test[img_id:img_id + 1, :, :, :]
        C = self.convout_f[n - 1]([X])
        C = np.squeeze([C])
        print("C"+str(n), " shape : ", C.shape)

        pl.figure(figsize=(15, 15))
        pl.suptitle('convout'+str(n))
        nice_imshow(pl.gca(), make_mosaic(C, 6, 6), cmap=cm.binary)
        pl.show()


Net = CNNetwork(WEIGHTS_FNAME="Mnist_Cnn_My_weights.hdf")

Net.get_DB()

# NewNetwork.init_model()

# NewNetwork.compile_model()

# NewNetwork.load_weights()

# NewNetwork.train_model(nb_epoch=1)

# NewNetwork.print_img()

# NewNetwork.visu_first_layer()

# NewNetwork.visu_weights()

# NewNetwork.visu_convo(1)
# NewNetwork.visu_convo(2)

# NewNetwork.predict_all()

# NewNetwork.get_score()

# NewNetwork.summary()

