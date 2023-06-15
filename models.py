from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, RepeatVector, GRU, LSTM, Bidirectional, TimeDistributed
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D

def lstm_model(input_shape,features,n_future):
    model = Sequential()
    model.add(LSTM(180, activation='relu', input_shape=input_shape, return_sequences=False))
    model.add(Dense(n_future*features))
    model.add(Reshape((n_future,features)))
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    return model

def gru_model(input_shape,features,n_future):
    model = Sequential()
    model.add(GRU(180, activation='relu', input_shape=input_shape, return_sequences=False))
    model.add(Dense(n_future*features))
    model.add(Reshape((n_future,features)))
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    return model

def bd_lstm_model(input_shape,features,n_future):
    model = Sequential()
    model.add(Bidirectional(LSTM(180, activation='relu', return_sequences=True, input_shape=input_shape)))
    # model.add(RepeatVector(n_future))
    model.add(Bidirectional(LSTM(180, activation='relu', return_sequences=True)))
    model.add(TimeDistributed(Dense(64, activation='relu'))) # For TimeDistributed see this https://stackoverflow.com/questions/45590240/lstm-and-cnn-valueerror-error-when-checking-target-expected-time-distributed
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    # model.add(Dense(features*n_future))
    # model.add(Reshape((features,n_future)))
    model.add(Dense(n_future*features))
    model.add(Reshape((n_future,features)))
    model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    return model

def cnn_lstm_model(input_shape,features,n_future):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(n_future))
    model.add(LSTM(180, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(n_future*features))
    model.add(Reshape((n_future,features)))
    model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    return model