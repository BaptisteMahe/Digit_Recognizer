import numpy as np
import os
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from time import time


class Network:

    def __init__(self):
        self.Bapp = None
        self.Xapp = None
        self.Yapp = None

        self.Bgen = None
        self.Xgen = None

        self.indices = None

        self.pred = None

        self.model = None

        self.WEIGHTS_FNAME = 'My_weights.hdf'

        self.training_time = None

    def get_Bapp(self, CSVapp):
        self.Bapp = pd.read_csv(CSVapp, header=0).values
        self.Xapp = self.Bapp[:, 1:]
        self.Yapp = self.Bapp[:, 0]

    def get_Bgen(self, CSVgen):
        self.Bgen = pd.read_csv(CSVgen, header=0).values
        self.Xgen = self.Bgen
        self.indices = np.array(range(1, len(self.Xgen) + 1))[:, np.newaxis]

    def init_model(self, layers, input_dim):
        self.model = Sequential()

        for i in range(len(layers)):
            if i == 0:
                self.model.add(Dense(layers[i], input_dim=input_dim, activation='relu'))
            else:
                self.model.add(Dense(layers[i], activation='relu'))

        # self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta',
                           metrics=['acc'])

    def load_weights(self):
        if os.path.exists(self.WEIGHTS_FNAME):
            print('Loading existing weights...')
            self.model.load_weights(self.WEIGHTS_FNAME)

    def train_model(self, epochs=1):
        t = time()
        self.model.fit(self.Xapp, to_categorical(self.Yapp), epochs=epochs)
        self.model.save_weights(self.WEIGHTS_FNAME)
        self.training_time = time() - t

    def print_training_time(self):
        print("\nTraining time = " + str(int(self.training_time*1000)/1000) + "s")

    def get_prediction(self):
        self.pred = self.model.predict_classes(self.Xgen)

    def generate_prediction_csv(self):
        preditions = self.pred[:, np.newaxis]
        np.savetxt("mespredictions.csv", np.c_[self.indices, preditions], delimiter=",", header="ImageId,Label",
                   comments="", fmt='%.1u')


NewNetwork = Network()

CSVgen = "test.csv"
CSVapp = "train.csv"
epochs = 5
input_dim = 784
layers = [64, 64]


NewNetwork.get_Bapp(CSVapp)
NewNetwork.get_Bgen(CSVgen)

NewNetwork.init_model(layers, input_dim)
NewNetwork.compile_model()

NewNetwork.load_weights()
NewNetwork.train_model(epochs)
NewNetwork.print_training_time()

NewNetwork.get_prediction()

NewNetwork.generate_prediction_csv()
