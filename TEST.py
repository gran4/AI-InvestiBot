import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Set the start and end dates for the historical data
start_date = '2020-01-01'
end_date = '2023-05-16'

# Define the stock symbol you want to retrieve data for
stock_symbol = 'AAPL'  # Replace with your desired stock symbol

# Use yfinance to fetch the stock data from Yahoo Finance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

num_days = 60#stock_data.size#['Close'].shape[0]  # Number of previous days' closing prices to consider

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create the training data sequences and labels
def create_sequences(data, num_days):
    sequences = []
    labels = []
    for i in range(num_days, len(data)):
        sequences.append(data[i-num_days:i, 0])
        labels.append(data[i, 0])
    return np.array(sequences), np.array(labels)

X_train, Y_train = create_sequences(train_data, num_days)
X_test, Y_test = create_sequences(test_data, num_days)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(num_days, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, batch_size=32, epochs=200)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the scaled data to get the actual prices
train_predictions = scaler.inverse_transform(train_predictions)
y_train = scaler.inverse_transform(Y_train.reshape(-1, 1)).flatten()
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()

print(y_train.shape[0], y_test.shape[0])

y_train_size = y_train.shape[0]
days = [i for i in range(y_train.shape[0])]
days2 = [i+y_train_size for i in range(y_test.shape[0])]
print(train_size, y_test.shape[0])
print(len(days), len(days2))

# Plot the actual and predicted prices
plt.figure(figsize=(12, 6))

predicted_test = plt.plot(days2, test_predictions, label='Predicted Test')
actual_test = plt.plot(days2, y_test, label='Actual Test')

#plt.plot(stock_data.index[num_days:num_days+len(train_predictions)], train_predictions, label='Predicted Train')
#plt.plot(stock_data.index[-len(test_predictions):], test_predictions, label='Predicted Test')

predicted_train = plt.plot(days, train_predictions, label='Predicted Train')
actual_train = plt.plot(days, y_train, label='Actual Train')


train_rmse = np.sqrt(np.mean((train_predictions - y_train)**2))
test_rmse = np.sqrt(np.mean((test_predictions - y_test)**2))
print('Train RMSE:', train_rmse)
print('Test RMSE:', test_rmse)

plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend([predicted_test[0], actual_test[0], predicted_train[0], actual_train[0]], ['Predicted Test', 'Actual Test', 'Predicted Train', 'Actual Train'])
plt.show()