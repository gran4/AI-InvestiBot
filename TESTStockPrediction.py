import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Set the start and end dates for the historical data
start_date = '2020-01-01'
end_date = '2023-06-09'

# Define the stock symbol you want to retrieve data for
stock_symbol = 'AAPL'  # Replace with your desired stock symbol

# Use yfinance to fetch the stock data from Yahoo Finance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

num_days = 60  # Number of previous days' closing prices to consider

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
        sequences.append(data[i - num_days:i, 0])
        labels.append(data[i, 0])
    return np.array(sequences), np.array(labels)

X_train, Y_train = create_sequences(train_data, num_days)
X_test, Y_test = create_sequences(test_data, num_days)

y_train_size = len(Y_train)
days_train = pd.date_range(start=start_date, periods=Y_train.shape[0])
days_test = pd.date_range(start=days_train[-1] + pd.DateOffset(1), periods=Y_test.shape[0])

# Plot the actual and predicted prices
plt.figure(figsize=(18, 6))

actual_train = plt.plot(days_train, Y_train, label='Actual Train')
actual_test = plt.plot(days_test, Y_test, label='Actual Test')

plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend([actual_test[0], actual_train[0]], ['Actual Test', 'Actual Train'])
plt.show()
