import json

import numpy as np
import pandas as pd

from typing import Optional, List
from sklearn.metrics import mean_squared_error
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
        other_vals['Dates'] = other_vals['Dates'][i:]
        for key in information_keys:
            if key in excluded_values:
                continue
            other_vals[key] = other_vals[key][i:]
    else:
        raise ValueError(f"Run getInfo.py with start date before {start_date}\n and end date after {start_date}")
    if end_date in other_vals['Dates']:
        i = other_vals['Dates'].index(end_date)
        other_vals['Dates'] = other_vals['Dates'][:i]
        for key in information_keys:
            if key in excluded_values:
                continue
            other_vals[key] = other_vals[key][:i]
    else:
        raise ValueError(f"Run getInfo.py with start date before {end_date}\n and end date after {end_date}")


    for key in information_keys:
        if pd.api.types.is_numeric_dtype(other_vals[key]):
            continue
        other_vals[key] = np.array(other_vals[key])


        # Calculate the minimum and maximum values
        min_value = np.min(other_vals[key])
        max_value = np.max(other_vals[key])


        if max_value - min_value != 0:
            # Scale the data
            other_vals[key] = (other_vals[key] - min_value) / (max_value - min_value)


    if "earnings diff" in information_keys:
        pass


    # Convert the dictionary of lists to a NumPy array
    filtered = [other_vals[key] for key in information_keys if key not in excluded_values]


    filtered = np.transpose(filtered)
    return filtered, start_date, end_date


class BaseModel():
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL", information_keys: List=[]) -> None:
        num_days = 60
        data, start_date, end_date = get_relavant_Values(start_date, end_date, stock_symbol, information_keys)
        shape = data.shape[1]

        # Split the data into training and testing sets
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

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

        # Calculate RMSSE for training and testing predictions
        train_rmsse = np.sqrt(mean_squared_error(Y_train, train_predictions)) / np.mean(Y_train[1:] - Y_train[:-1])
        test_rmsse = np.sqrt(mean_squared_error(Y_test, test_predictions)) / np.mean(Y_test[1:] - Y_test[:-1])

        print('Train RMSSE:', train_rmsse)
        print('Test RMSSE:', test_rmsse)


class DayTradeModel(BaseModel):
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close']
        )


class MACDModel(BaseModel):
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close', 'MACD', 'Histogram', 'flips', '200-day EMA']
        )


class ImpulseMACDModel(BaseModel):
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close', 'Histogram', 'Momentum', 'Change', 'flips', '200-day EMA']
        )


class ReversalModel(BaseModel):
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close', 'gradual-liquidity spike', '3-liquidity spike', 'momentum_oscillator']
        )


class MiscModel(BaseModel):
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close', '3-liquidity spike', 'earnings dates', 'earnings diff', 'Histogram', 'flips']
        )


DayTradeModel()
MACDModel()
ImpulseMACDModel()
ReversalModel()
#MiscModel()