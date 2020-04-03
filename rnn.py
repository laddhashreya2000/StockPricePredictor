import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
#we cannot do training_set = dataset_train.iloc[:, 1].values as this will create a vector and not a numpy array.
training_set = dataset_train.iloc[:, 1:2].values

# apply normalisation for sigmoid fn
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# at each time t, the rnn is going to look at 60 previous time steps(including t) and then predict t+1 value. This is experimental.
# 60 timesteps correspond to three months.
X_train = []
y_train = []
for i in range(60, 1258):# we start at index 0 and end at 1257.
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
#convert to array so can be used by rnn
X_train, y_train = np.array(X_train), np.array(y_train)

#adding more dimensionality to data structure, we can add other indicators using this new dimensionality.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#last 1 is no. of indicators which is the open stock price here.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
# we want high dimensionality so 50 neurons in one layer.
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.1))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.1))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.1))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.1))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.1))
regressor.add(Dense(units = 1))

#experiment different optimizers like rmsprop, mse because its a regression.
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
#loss should be close for last 10 epochs to avoid overfitting
#we can do the grid search here also, scoring = 'neg_mean_squared_error'

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) #vertical axis so axis=0
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)#makes it in one column
inputs = sc.transform(inputs)#no fit here
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
