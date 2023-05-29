import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Set the start and end dates for the historical data
start_date = '2010-01-01'
end_date = '2023-05-16'

# Define the stock symbol you want to retrieve data for
stock_symbol = 'AAPL'  # Replace with your desired stock symbol

# Use yfinance to fetch the stock data from Yahoo Finance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create the training data sequences and labels
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(sequence_length, len(data)):
        sequences.append(data[i-sequence_length:i, 0])
        labels.append(data[i, 0])
    return np.array(sequences), np.array(labels)

sequence_length = 60  # Number of previous days' closing prices to consider
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the scaled data to get the actual prices
train_predictions = scaler.inverse_transform(train_predictions)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Plot the actual and predicted prices
plt.figure(figsize=(12, 6))
train_plot = plt.plot(x_train, y_train, label='Actual Train')
test_plot = plt.plot(x_test, y_test, label='Actual Test')

#plt.plot(stock_data.index[sequence_length:sequence_length+len(train_predictions)], train_predictions, label='Predicted Train')
#plt.plot(stock_data.index[-len(test_predictions):], test_predictions, label='Predicted Test')

plt.plot(X_train, test_predictions, label='Predicted Test')
plt.plot(X_test, test_predictions, label='Predicted Test')

plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend([train_plot[0], test_plot[0]], ['Actual Train', 'Actual Test'])
plt.show()