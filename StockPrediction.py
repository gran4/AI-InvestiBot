import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
#import alpha_vantage
#from alpha_vantage.techindicators import TechIndicators


"""
#load data
company = 'APPL'

start = dt.datetime(2015, 1, 1)
end = dt.datetime(2018, 3, 3)

data = web.DataReader(company, 'yahoo', start, end) 
"""

# Set the start and end dates for the historical data
start_date = '2010-01-01'
end_date = '2023-05-16'

# Define the stock symbol you want to retrieve data for
stock_symbol = 'AAPL'  # Replace with your desired stock symbol

# Use yfinance to fetch the stock data from Yahoo Finance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)


stock_data_index = stock_data['Close'].index
stock_data['Close'] = stock_data['Close'].iloc[::-1]
stock_data['Close'].index = stock_data_index


# Calculate the MACD using pandas' rolling mean functions
stock_data['12-day EMA'] = stock_data['Close'].ewm(span=12).mean()
stock_data['26-day EMA'] = stock_data['Close'].ewm(span=26).mean()
stock_data['MACD'] = stock_data['12-day EMA'] - stock_data['26-day EMA']
stock_data['Signal Line'] = stock_data['MACD'].ewm(span=9).mean()


scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(stock_data['Close'])





#MAKE DATA USABLE
prediction_range = 60

x_train = []
y_train = []

for x in range(prediction_range, len(stock_data)):
    x_train.append(scaled_data[x-prediction_range: x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#make it 3d bc 3 in Tuple
#x_train = np.reshape(x_train, (x_train.shape[0], x_train[1], 1))





#BUILD MODEL
model = Sequential()

#add layers to model
model.add(LSTM(units = 50, return_sequences = True, input_shape = x_train.shape[1][1]))
model.add(Dropout(.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(.2))

model.add(LSTM(units = 50))
model.add(Dropout(.2))


#END With DENSE Layer
model.add(Dense(units = 1))#final prediction of the closing price



#compile to use
model.compile(optimizer="adam", loss="mean_squared_error")

#TRAIN MODEL
model.fit(x_train, y_train, epochs=25, batch_size=32)






#TEST THE MODEL ACCURACY

#LOAD IT
test_start = dt.datetime(2023, 3, 3)
test_end = dt.datetimenow()

# Use yfinance to fetch the stock data from Yahoo Finance
test_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Calculate the MACD using pandas' rolling mean functions
stock_data['12-day EMA'] = stock_data['Close'].ewm(span=12).mean()
stock_data['26-day EMA'] = stock_data['Close'].ewm(span=26).mean()
stock_data['MACD'] = stock_data['12-day EMA'] - stock_data['26-day EMA']
stock_data['Signal Line'] = stock_data['MACD'].ewm(span=9).mean()

actual_prices = test_data["Close"]


total_dataset = pd.concat((stock_data["Close"], test_data["Close"]))

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_range:]
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)


#MAKE PREDICTIONS
x_test = []

for x in range(prediction_range, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_range:x, 0])

x_test = np.array(x_test)
#x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


#PLOT TEST PREDICTIONS
plt.plot(actual_prices, color = "black", label = f"Actual {stock_symbol} price")
plt.plot(predicted_prices, color = "green", label = f"Predicted {stock_symbol} price")

plt.title(f"{stock_symbol} Share Price")

plt.xlabel("Time")
plt.ylabel(f"{stock_symbol} Share Price")

plt.legend()
plt.show

