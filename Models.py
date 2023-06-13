import json

import numpy as np
import pandas as pd
import yfinance as yf

from typing import Optional, List
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pandas_market_calendars import get_calendar

from Tradingfuncs import create_sequences



excluded_values = (
    "earnings dates",
    "earnings diff"
)


def get_relavant_Values(start_date: str, end_date: str, stock_symbol: str, information_keys: List[str]) -> np.array:
    """Returns information asked for and corrected dates"""    
    #Check if start_day is a holiday
    nyse = get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    start_date = pd.to_datetime(start_date).date()
    if start_date not in schedule.index:
        # Find the next trading day
        next_trading_day = nyse.valid_days(start_date=start_date, end_date=end_date)[0]
        start_date = next_trading_day.date().strftime('%Y-%m-%d')

    end_date = pd.to_datetime(end_date).date()
    if end_date not in schedule.index:
        end_date = schedule.index[-1].date().strftime('%Y-%m-%d')


    with open(f'{stock_symbol}/info.json') as file:
        other_vals = json.load(file)
    if start_date in other_vals['Dates']:
        i = other_vals['Dates'].index(start_date)
        for key in information_keys:
            vals = other_vals[key]
            vals = vals[i:]
    else:
        raise ValueError(f"Run getInfo.py with start date before {start_date}")

    if end_date in other_vals['Dates']:
        i = other_vals.index(end_date)
        for key in information_keys:
            vals = other_vals[key]
            vals = vals[i:]
    else:
        raise ValueError(f"Run getInfo.py with end date after {end_date}")
        
    # Convert the dictionary of lists to a NumPy array
    other_vals = np.array(list(other_vals.values())).T
    return other_vals, start_date, end_date


class BaseModel():
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-09",
                 stock_symbol: str = "AAPL", information_keys: List=[]) -> None:
        other_vals = get_relavant_Values(start_date, end_date, stock_symbol, information_keys)

        # Concatenate the closing prices array and other_data_arr along axis 1 (column-wise concatenation)
        close_vals = close_vals[:, np.newaxis]
        close_vals = close_vals.squeeze()


        total_vals = np.concatenate((close_vals, other_vals), axis = 1)
        shape = len(total_vals)
        scaled_data = scaler.fit_transform(total_vals)

        # Split the data into training and testing sets
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]


        X_total, Y_total = create_sequences(train_data, num_days)
        X_train, Y_train = create_sequences(train_data, num_days)
        X_test, Y_test = create_sequences(test_data, num_days)

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(num_days, shape)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_total, Y_total, batch_size=32, epochs=20)
        model.fit(X_test, Y_test, batch_size=32, epochs=20)
        model.fit(X_train, Y_train, batch_size=32, epochs=20)

        # Save structure to json
        jsonversion = model.to_json()
        with open(f"{stock_symbol}/model.json", "w") as json_file:
            json_file.write(jsonversion)

        # Save weights to HDF5
        model.save_weights(f"{stock_symbol}/weights.h5")

        # Make predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # Inverse transform the scaled data to get the actual prices
        train_predictions = scaler.inverse_transform(train_predictions)
        y_train = scaler.inverse_transform(Y_train.reshape(-1, 1)).flatten()
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()

        train_rmse = np.sqrt(np.mean((train_predictions - y_train)**2))
        test_rmse = np.sqrt(np.mean((test_predictions - y_test)**2))
        print('Train RMSE:', train_rmse)
        print('Test RMSE:', test_rmse)


class DayTradeModel(BaseModel):
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-09",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close']
        )


class MACDModel(BaseModel):
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-09",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close', 'MACD', 'Histogram', 'flips', '200-day EMA']
        )


class ImpulseMACDModel(BaseModel):
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-09",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close', 'Histogram', 'Momentum', 'Change', 'flips', '200-day EMA']
        )


class ReversalModel(BaseModel):
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-09",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close', 'gradual-liquidity spike', '4-liquidity spike', 'momentum_oscillator']
        )


class MiscModel(BaseModel):
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-09",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close', '4-liquidity spike', 'earnings dates', 'earnings diff', 'Histogram', 'flips']
        )


MACDModel()