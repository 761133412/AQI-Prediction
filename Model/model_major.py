from keras.models import Input, Model, Sequential                     #### TCN
from keras.layers import Dense, Activation, Conv1D, LSTM, Dropout, Reshape, Bidirectional, Flatten, Add, Concatenate, MaxPool1D, LeakyReLU
from keras.callbacks import Callback
from Support.support_tcn import TCN

from Support.support_nested_lstm import NestedLSTM      ##### NLSTM
from Support.support_OL import ONLSTM


import time


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.time()

    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def buildTCN_LSTM(timestep):

    batch_size, timesteps, input_dim = None, timestep, 1
    i = Input(batch_shape=(batch_size, timesteps, input_dim))
    x = TCN(return_sequences=False)(i)  # The TCN layers are here.
    #x = Dense(32)(x)

    x = Reshape((-1, 1))(x)
    x = LSTM(output_dim=100, return_sequences=True)(x)
    print(x.shape)
    x = LSTM(100)(x)
    x = Dropout(0.2)(x)

    o = Dense(1)(x)
    o = Activation('linear')(o)
    #output_layer = x
    model = Model(inputs=[i], outputs=[o])
    # model.summary()
    model.compile(optimizer='rmsprop', loss='mse',)

    return model


def build_LSTM(time_step):
    model = Sequential()            # layers = [1, ahead_, 100, 1]
    model.add(LSTM(
            input_shape=(time_step, 1),
            output_dim=time_step,
            # units=layers[2],
            return_sequences=True))
        #model.add(Dropout(0.2))
    model.add(LSTM(
            100,
            return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(
            output_dim=1))
    model.add(Activation("linear"))

    model.summary()
    model.compile(loss="mse", optimizer="rmsprop")
    return model


def buildNLSTM(timestep):

    batch_size, timesteps, input_dim = None, timestep, 1
    i = Input(batch_shape=(batch_size, timesteps, input_dim))

    # x = Reshape((-1, 1, 1))(i)
    x = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(i)
    x = Dense(16, activation='linear')(x)

    # x = Dense(64)(i)
    # x = Bidirectional(NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1))(x)
    print(x.shape)
    # x = LSTM(100)(x)

    # o = Dense(1)(x)
    # o = Activation('linear')(o)
    o = Dense(1, activation="linear")(x)
    # output_layer = x
    # model = Sequential()
    model = Model(inputs=[i], outputs=[o])

    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()

    return model


def buildBLSTM(timestep):

    batch_size, timesteps, input_dim = None, timestep, 1
    i = Input(batch_shape=(batch_size, timesteps, input_dim))

    # x = Reshape((-1, 1, 1))(i)
    # x = NestedLSTM(64, depth=2, dropout=0.0, recurrent_dropout=0.1)(i)

    # x = Dense(64)(i)
    # x = Bidirectional(LSTM(output_dim=64, return_sequences=True))(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(i)
    x = Dropout(0.2)(x)
    print(x.shape)
    # x = LSTM(100)(x)
    x = Flatten()(x)
    o = Dense(1)(x)
    o = Activation('linear')(o)
    # output_layer = x
    # model = Sequential()
    model = Model(inputs=[i], outputs=[o])

    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()

    return model


def buildSLSTM(timestep):
    batch_size, timesteps, input_dim = None, timestep, 1
    i = Input(batch_shape=(batch_size, timesteps, input_dim))

    x = Reshape((-1, 1))(i)
    x = LSTM(output_dim=128, return_sequences=True)(x)
    x = LSTM(output_dim=64, return_sequences=True)(x)
    x = LSTM(output_dim=64, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)

    o = Dense(1)(x)
    o = Activation('linear')(o)
    # output_layer = x
    model = Model(inputs=[i], outputs=[o])

    model.compile(optimizer='rmsprop', loss='mse', )
    model.summary()
    return model
