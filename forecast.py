import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from math import sqrt
import argparse
import json
import models
import time
import sys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-c", "--configFile", type=str, help="path and name to the configuration file")
args = vars(ap.parse_args())

params = json.loads(open(args["configFile"]).read())
EPOCHS = params["epochs"]
BATCH_SIZE = params["batch_size"]
TRAIN_SIZE = params["train_size"]
TIMESTAMP_COLUMN = params["timestamp_column"]
SCALER = params["scaler"]
N_FUTURE = params["num_future_samples"]   # Number of samples we want to forecast into the future (Out).
N_PAST = params["num_past_samples"]   # Number of past samples we want to use for training (Step).
X = params["X"] # columns that will be used for training
Y = params["Y"] # columns that will be forecasted
features = len(Y) # Number of features
df = pd.read_csv(args["dataset"])
if TIMESTAMP_COLUMN == "None":
    print("[INFO] No timestamp related column is indicated.")
elif TIMESTAMP_COLUMN not in df.columns:
    print(f'[ERROR] Column "{TIMESTAMP_COLUMN}" does not exist in the DataFrame. No timestamp related column will be removed. If there is a timestamp related column, please type its name in the configuration file in the field named "timestamp_column"')
    time.sleep(9)
    print(f'[INFO] If there is no timestamp related column in the dataset, then you can type "None" in the field named "timestamp_column"')
    time.sleep(5)
    print("[INFO] The program will now exit...")
    time.sleep(2)
    sys.exit()
else:
    df = df.drop(TIMESTAMP_COLUMN, axis=1)

def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]

X_column_index = column_index(df, X)
Y_column_index = column_index(df, Y)

if SCALER == "Standard":
    SCALER = StandardScaler()
elif SCALER == "MinMax":
    SCALER = MinMaxScaler()
else:
    print(f'[ERROR] You can choose only between "MinMax" and "Standard" scaler for the data.')
    time.sleep(5)
    print("[INFO] The program will now exit...")
    time.sleep(2)
    sys.exit()


scaled_df = SCALER.fit_transform(df)


train_size = int(len(scaled_df) * TRAIN_SIZE)
test_size = len(scaled_df) - train_size
train, test = scaled_df[0:train_size,:], scaled_df[train_size:len(scaled_df),:]

def split_sequence(seq, steps, out):
    X, Y = list(), list()
    for i in range(len(seq)):
        end = i + steps
        outi = end + out
        if outi > len(seq)-1:
            break
        seqx, seqy = seq[i:end], seq[end:outi]
        X.append(seqx)
        Y.append(seqy)
    return np.array(X), np.array(Y)


# split into samples
X_train, Y_train = split_sequence(train, N_PAST, N_FUTURE)
X_test, Y_test = split_sequence(test, N_PAST, N_FUTURE)
X_train = np.take(X_train, X_column_index, X_train.ndim-1)
Y_train = np.take(Y_train, Y_column_index, Y_train.ndim-1)

X_test = np.take(X_test, X_column_index, X_test.ndim-1)
Y_test = np.take(Y_test, Y_column_index, Y_test.ndim-1)

model = models.cnn_lstm_model((X_train.shape[1], X_train.shape[2]),features,N_FUTURE)
print(model.summary())
time.sleep(5)
history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test),verbose=1, shuffle=False)

yhat = model.predict(X_test)

y_pred_future = SCALER.inverse_transform(yhat.reshape(yhat.shape[0]*yhat.shape[2], yhat.shape[1]))
y_actual = SCALER.inverse_transform(Y_test.reshape(Y_test.shape[0]*Y_test.shape[2], Y_test.shape[1]))#[:,1]

pyplot.plot(history.history['loss'], label='train', color="black")
pyplot.plot(history.history['val_loss'], label='test',marker='.')
pyplot.title('model loss',size=15)
pyplot.ylabel('loss',size=15)
pyplot.xlabel('epochs',size=15)
pyplot.legend(loc='upper right',fontsize=15)

pyplot.savefig("train-test_loss.pdf")

mae = mean_absolute_error(y_actual, y_pred_future)
print('Test Score: %.2f MAE' % (mae))
mse = mean_squared_error(y_actual, y_pred_future)
print('Test Score: %.2f MSE' % (mse))
rmse = sqrt(mse)
print('Test Score: %.2f RMSE' % (rmse))
r2 = r2_score(y_actual, y_pred_future)
print('Test Score: %.2f R2' % (r2))

