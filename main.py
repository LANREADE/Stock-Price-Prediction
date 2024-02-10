import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from pandas_datareader import data as pdr
import datetime as dt

# Downloading data for a particular company
yf.pdr_override()  # Overriding pandas datareader with Yahoo Finance
start_date = dt.datetime(2012, 1, 2)
end_date = dt.datetime(2019, 1, 2)
company = "FB"
data = pdr.get_data_yahoo(company, start=start_date, end=end_date)

# Preparing data for the neural network
scaler = MinMaxScaler(feature_range=(0, 1))
# Explicitly specify the number of rows (assuming the number of columns is 1)
scaled_data = scaler.fit_transform(data['Close'].values.reshape(len(data['Close']), 1))



prediction_days = 60

x_train = []
y_train = []

for i in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[i - prediction_days:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next closing value
model.compile(optimizer='adam', loss="mean_squared_error")
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Testing the model accuracy on unseen data
test_start_date = dt.datetime(2020, 1, 1)
test_end_date = dt.datetime.now()
test_data = pdr.get_data_yahoo(company, start=test_start_date, end=test_end_date)

actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []
for i in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[i - prediction_days:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plotting the test predictions
plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color='green', label=f"Predicted {company} Price")
plt.title(f"{company} Share Price Prediction")
plt.xlabel("Time")
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()
