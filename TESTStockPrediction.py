import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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
model.add(Dropout(.2))

model.add(LSTM(50))
model.add(Dropout(.2))


model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the scaled data to get the actual prices
train_predictions = scaler.inverse_transform(train_predictions)
y_train = scaler.inverse_transform([y_train])
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform([y_test])

# Evaluate the model
train_rmse = np.sqrt(np.mean((train_predictions - y_train)**2))
test_rmse = np.sqrt(np.mean((test_predictions - y_test)**2))
print('Train RMSE:', train_rmse)
print('Test RMSE:', test_rmse)
