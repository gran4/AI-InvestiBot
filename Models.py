import json

import numpy as np
import pandas as pd

from typing import Optional, List
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pandas_market_calendars import get_calendar
from datetime import datetime, timedelta

from Tradingfuncs import create_sequences, excluded_values


def process_earnings(dates: List, diffs: List, start_date: str, end_date: str):
    """
    Returns earnings bettween date range and filled to fit other vals.

    Gets earnings between start and end data
    then fills in 0s for dates without an earnings report
    """
    #_________________deletes earnings before start and after end______________________#
    delete_start = 0
    delete_end = 0
    for day in dates:
        if day < start_date:
            delete_start += 1
        elif day > end_date:
            delete_end += 1
    dates = dates[delete_start:]
    diffs = diffs[delete_start:]

    dates = dates[:delete_end]
    diffs = diffs[:delete_end]


    #_________________Fill Data out with 0s______________________#
    filled_dates = []
    filled_earnings = []

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Fill out the list to match the start and end date
    dates = [datetime.strptime(date_str, "%b %d, %Y") for date_str in dates]
    current_date = start_date
    while current_date <= end_date:
        filled_dates.append(current_date)
        if current_date in dates:
            existing_index = dates.index(current_date)
            filled_earnings.append(diffs[existing_index])
        else:
            filled_earnings.append(0)
        current_date += timedelta(days=1)
    return dates, diffs


def get_relavant_Values(start_date: str, end_date: str, stock_symbol: str, information_keys: List[str]) -> np.array:
    """Returns information asked for and corrected dates"""    
    #_________________Check if start or end is holiday______________________#
    nyse = get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    #_________________Change if it is a holiday______________________#
    start_date = pd.to_datetime(start_date).date()
    if start_date not in schedule.index:
        # Find the next trading day
        next_trading_day = nyse.valid_days(start_date=start_date, end_date=end_date)[0]
        start_date = next_trading_day.date().strftime('%Y-%m-%d')

    end_date = pd.to_datetime(end_date).date()
    if end_date not in schedule.index:
        end_date = schedule.index[-1].date().strftime('%Y-%m-%d')


    #_________________Load info______________________#
    with open(f'{stock_symbol}/info.json') as file:
        other_vals = json.load(file)

    #_________________Make Data fit between start and end date______________________#
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

    #_________________Process earnings______________________#
    if "earnings diff" in information_keys:
        dates = other_vals['earnings dates']
        diffs = other_vals['earnings diff']

        dates, diffs = process_earnings(dates, diffs, start_date, end_date)
        other_vals['earnings dates'] = dates
        other_vals['earnings diff'] = diffs

    #_________________Scales Data______________________#
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


    # Convert the dictionary of lists to a NumPy array
    filtered = [other_vals[key] for key in information_keys if key not in excluded_values]

    filtered = np.transpose(filtered)
    return filtered, start_date, end_date


class BaseModel():
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL", information_keys: List=[]) -> None:
        #_________________ GET Data______________________#
        num_days = 60
        data, start_date, end_date = get_relavant_Values(start_date, end_date, stock_symbol, information_keys)
        shape = data.shape[1]


        #_________________Process Data for LSTM______________________#
        # Split the data into training and testing sets
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

        X_total, Y_total = create_sequences(train_data, num_days)
        X_train, Y_train = create_sequences(train_data, num_days)
        X_test, Y_test = create_sequences(test_data, num_days)


        #_________________Train it______________________#
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

        #_________________Save Model______________________#
        # Save structure to json
        jsonversion = model.to_json()
        with open(f"{stock_symbol}/model.json", "w") as json_file:
            json_file.write(jsonversion)

        # Save weights to HDF5
        model.save_weights(f"{stock_symbol}/weights.h5")


        #_________________TEST QUALITY______________________#
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


class EarningsModel(BaseModel):
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close', 'earnings dates', 'earnings diff', 'Momentum']
        )



class MiscModel(BaseModel):
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close', '3-liquidity spike', 'earnings dates', 'earnings diff', 'Histogram', 'flips', 'momentum_oscillator']
        )


#DayTradeModel()
#MACDModel()
#ImpulseMACDModel()
#ReversalModel()
#EarningsModel()
#MiscModel()