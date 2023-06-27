import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Set the start and end dates for the historical data
#start_date = '2020-01-01'
#end_date = '2023-06-09'

# Define the stock symbol you want to retrieve data for
stock_symbol = 'AAPL'  # Replace with your desired stock symbol

# Use yfinance to fetch the stock data from Yahoo Finance
#stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

data = pd.read_csv(f'{stock_symbol}.csv')
X = data['Close']
y = data.drop(['Close'], axis=1)

# Train test spit
x_train, x_test, y_train, y_test = train_test_split(X, y)

# Create the sequential
network = Sequential()

y_train_size = y_train.shape[0]
days_train = [i for i in range(y_train_size)]
days_test = [i+y_train_size for i in range(y_train_size)]



# Build the LSTM model
model = Sequential()
model.add(Dense(1, input_shape=(1, )))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


# Train the model
network.fit(x_train.values, y_train.values, batch_size=32, epochs=100)


# Plot the actual and predicted prices
plt.figure(figsize=(18, 6))

actual_train = plt.plot(x_train.values, y_train.values, label='Actual Train')
actual_test = plt.plot(x_test.values, y_test.values, label='Actual Test')

plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend([actual_test[0], actual_train[0]], ['Actual Test', 'Actual Train'])
plt.show()
