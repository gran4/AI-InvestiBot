"""
Name:
    Models.py

Purpose:
    This module provides the classes for all the models which can be trained and used to 
    predict stock prices. The models themselves all inherit the methods from the BaseModel 
    with variations in symbols and information keys etc.

Author:
    Grant Yul Hur

See also:
    Other modules related to running the stock bot -> lambda_implementation, loop_implementation
"""

import json

from typing import Optional, List, Dict
from typing_extensions import Self
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

from trading_funcs import get_relavant_values, create_sequences, process_flips
from getInfo import calculate_momentum_oscillator, get_liquidity_spikes, get_earnings_history
from warnings import warn
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


class BaseModel:
    """
    This is the base class for all the models. It handles the actual training, saving, loading, predicting, etc.
    Setting the `information_keys` allows us to describe what the model uses. The information keys themselves
    are retrieved from a json format that was created by getInfo.py.

    Args:
        start_date (str): The start date of the training data
        end_date (str): The end date of the training data
        stock_symbol (str): The stock symbol of the stock you want to train on
        num_days (int): The number of days to use for the LSTM model
        information_keys (List[str]): The information keys that describe what the model uses
    """
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL",
                 num_days: int = 60,
                 information_keys: List[str]=["Close"]) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.stock_symbol = stock_symbol
        self.information_keys = information_keys
        self.num_days = num_days

        self.cached: Optional[pd.DataFrame] = None
        self.model: Optional[Sequential] = None
        self.data: Optional[Dict] = None

#________For testing with offline predicting____________#
        self.cached_json: Optional[Dict] = None

    def train(self, epochs=100):
        """
        Trains Model off `information_keys`

        Args:
            epochs (int): The number of epochs to train the model for

        Returns:
            None
        """
        warn("If you saved before, use load func instead")

        start_date = self.start_date
        end_date = self.end_date
        stock_symbol = self.stock_symbol
        information_keys = self.information_keys
        num_days = self.num_days

        #_________________ GET Data______________________#
        data, start_date, end_date = get_relavant_values(start_date, end_date, stock_symbol, information_keys)
        shape = data.shape[1]

        temp = {}
        for key, val in zip(information_keys, data):
            temp[key] = list(val)
        self.data = temp


        #_________________Process Data for LSTM______________________#
        # Split the data into training and testing sets
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

        x_total, y_total = create_sequences(train_data, num_days)
        x_train, y_train = create_sequences(train_data, num_days)
        x_test, y_test = create_sequences(test_data, num_days)


        #_________________Train it______________________#
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(num_days, shape)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(x_test, y_test, batch_size=32, epochs=epochs)
        model.fit(x_train, y_train, batch_size=32, epochs=epochs)
        model.fit(x_total, y_total, batch_size=32, epochs=epochs)

        self.model = model

    def save(self) -> None:
        """
        This method will save the model using the tensorflow save method. It will also save the data
        into the `json` file format.

        Args:
            None
        
        Returns:
            None
        """
        #_________________Save Model______________________#
        self.model.save(f"{self.stock_symbol}/model")

        with open(f"{self.stock_symbol}/data.json", "w") as json_file:
            json.dump(self.data, json_file)

    def test(self) -> None:
        """
        A method for testing purposes. It's EXPENSIVE and should only be used for testing purposes.

        Args:
            None
        
        Returns:
            None
        """
        warn("Expensive, for testing purposes")

        if not self.model:
            raise LookupError("Compile or load model first")

        start_date = self.start_date
        end_date = self.end_date
        stock_symbol = self.stock_symbol
        information_keys = self.information_keys
        num_days = self.num_days

        #_________________ GET Data______________________#
        data, start_date, end_date = get_relavant_values(start_date, end_date, stock_symbol, information_keys)

        #_________________Process Data for LSTM______________________#
        # Split the data into training and testing sets
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]


        x_train, y_train = create_sequences(train_data, num_days)
        x_test, y_test = create_sequences(test_data, num_days)
        #_________________TEST QUALITY______________________#
        train_predictions = self.model.predict(x_train)
        test_predictions = self.model.predict(x_test)

        train_data = train_data[num_days:]
        test_data = test_data[num_days:]
        

        #Get first collumn
        temp_train = train_data[:, 0]
        temp_test = test_data[:, 0]


        # Calculate RMSSE for training predictions
        train_rmse = np.sqrt(mean_squared_error(temp_train, train_predictions))
        train_abs_diff = np.mean(np.abs(train_data[1:] - train_data[:-1]))
        train_rmsse = train_rmse / train_abs_diff

        # Calculate RMSSE for testing predictions
        test_rmse = np.sqrt(mean_squared_error(temp_test, test_predictions))
        test_abs_diff = np.mean(np.abs(test_data[1:] - test_data[:-1]))
        test_rmsse = test_rmse / test_abs_diff

        print('Train RMSE:', train_rmse)
        print('Test RMSE:', test_rmse)
        print()
        print('Train RMSSE:', train_rmsse)
        print('Test RMSSE:', test_rmsse)
        print()
        def is_homogeneous(arr):
            return len(set(arr.dtype for arr in arr.flatten())) == 1
        print("Homogeneous(Should be True):", is_homogeneous(data))


        y_train_size = y_train.shape[0]
        days_test = [i for i in range(y_train.shape[0])]
        days_train = [i+y_train_size for i in range(y_test.shape[0])]

        # Plot the actual and predicted prices
        plt.figure(figsize=(18, 6))

        predicted_train = plt.plot(days_test, train_predictions, label='Predicted Train')
        actual_train = plt.plot(days_test, y_train, label='Actual Train')

        predicted_test = plt.plot(days_train, test_predictions, label='Predicted Test')
        actual_test = plt.plot(days_train, y_test, label='Actual Test')


        plt.title(f'{stock_symbol} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend([predicted_test[0], actual_test[0], predicted_train[0], actual_train[0]], ['Predicted Test', 'Actual Test', 'Predicted Train', 'Actual Train'])
        plt.show()

    def load(self) -> Optional[Self]:
        """
        This method will load the model using the tensorflow load method.
        
        Args:
            None

        Returns:
            self.model (Optional[Self]]): The model that was loaded
        """
        if self.model:
            return
        self.model = load_model(f"{self.stock_symbol}/model")

        with open(f"{self.stock_symbol}/data.json") as file:
            self.data = json.load(file)

        return self.model

    def update_cached_online(self):
        """
        This method will update the cached data online.

        Args:
            None

        Returns:
            None
        """
        cached_data = self.cached
        #_________________ GET Data______________________#
        ticker = yf.Ticker(self.stock_symbol)
        if cached_data:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            day_data = ticker.history(start=end_date, end=end_date, interval="1d")
            cached_data = cached_data.drop(cached_data.index[0])
            cached_data.append(day_data, ignore_index=True)
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = end_date - timedelta(days=260)
            cached_data = ticker.history(start=start_date, end=end_date, interval="1d")
        if not len(cached_data):
            raise ConnectionError("Stock data failed to load. Check your internet")

    def update_cached_offline(self):
        """
        This method will update the cached data offline.

        Args:
            None
        
        Methods:
            None
        """
        start_date = self.start_date
        end_date = self.end_date
        cached_data = self.cached
        #_________________ GET Data______________________#
        if not self.cached_json:
            with open(f"{self.stock_symbol}/data.json") as file:
                self.cached_json = json.load(file)
                self.cached = pd.DataFrame.from_dict(self.cached_json)
                cached_data = {}
            if not start_date in cached_data['Dates']:
                raise ValueError("End is before or after `Dates` range")
        i_end = cached_data['Dates'].index(end_date)
        if cached_data:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            day_data = {key: self.cached_json[key][i_end] for key in self.information_keys}
            day_data["Dates"] = self.cached_json['Dates'][i_end]

            cached_data = cached_data.drop(cached_data.index[0])
            cached_data.append(day_data, ignore_index=True)
        if not len(cached_data):
            raise ConnectionError("Stock data failed to load. Check your internet")

    def getInfoToday(self, period: int=14) -> List[float]:
        """
        This method will get the information for the stock today and the last relevant days to the stock.
        
        The cached_data is used so less data has to be retrieved from yf.finance as it is already being
        held and is quicker to obtain.

        Args:
            period (int): The number of days to get the information for
        
        Returns:
            stock_data[key].values.tolist() (List[float]): The information for the stock today and the last relevant days to the stock
        """
        #Limit attribute look ups + Improve readability
        stock_symbol = self.stock_symbol
        information_keys = self.information_keys
        end_date = self.end_date
        num_days = self.num_days
        cached_data = self.cached
        if not cached_data:
            try:
                self.update_cached_online()
            except ConnectionError:
                warn("Stock data failed to download. Check your internet")
                self.update_cached_offline()

        stock_data = {}

        date_object = datetime.strptime(self.end_date, "%Y-%m-%d")
        next_day = date_object + timedelta(days=1)
        self.end_date = next_day.strftime("%Y-%m-%d")

        stock_data['Close'] = cached_data['Close']
        #_________________MACD Data______________________#
        # Calculate start and end dates
        stock_data['12-day EMA'] = cached_data['Close'].ewm(span=12, adjust=False).mean().iloc[-60:]

        # 26-day EMA
        stock_data['26-day EMA'] = cached_data['Close'].ewm(span=26, adjust=False).mean().iloc[-60:]

        # MACD
        stock_data['MACD'] = stock_data['12-day EMA'] - stock_data['26-day EMA']

        # Signal Line
        span = 9
        signal_line = stock_data['MACD'].rolling(window=span).mean().iloc[-60:]
        stock_data['Signal Line'] = signal_line
        stock_data['Histogram'] = stock_data['MACD'] - stock_data['Signal Line']
        stock_data['200-day EMA'] = cached_data['Close'].ewm(span=200, adjust=False).mean().iloc[-60:]
        stock_data['Change'] = cached_data['Close'].diff().iloc[-60:]
        stock_data['Momentum'] = stock_data['Change'].rolling(window=10, min_periods=1).sum().iloc[-60:]
        stock_data["Gain"] = stock_data["Change"].apply(lambda x: x if x > 0 else 0)
        stock_data['Loss'] = stock_data['Change'].apply(lambda x: abs(x) if x < 0 else 0)
        stock_data["Avg Gain"] = stock_data["Gain"].rolling(window=14, min_periods=1).mean().iloc[-60:]
        stock_data["Avg Loss"] = stock_data["Loss"].rolling(window=14, min_periods=1).mean().iloc[-60:]
        stock_data["RS"] = stock_data["Avg Gain"] / stock_data["Avg Loss"]
        stock_data["RSI"] = 100 - (100 / (1 + stock_data["RS"]))

        # TRAMA
        volatility = cached_data['Close'].diff().abs().iloc[-60:]
        trama = cached_data['Close'].rolling(window=period).mean().iloc[-60:]
        stock_data['TRAMA'] = trama + (volatility * 0.1)

        # Reversal
        stock_data['gradual-liquidity spike'] = get_liquidity_spikes(cached_data['Volume'], gradual=True).iloc[-60:]
        stock_data['3-liquidity spike'] = get_liquidity_spikes(cached_data['Volume'], z_score_threshold=4).iloc[-60:]
        stock_data['momentum_oscillator'] = calculate_momentum_oscillator(cached_data['Close']).iloc[-60:]


        #_________________12 and 26 day Ema flips______________________#
        temp = []
        ema12=cached_data['Close'].ewm(span=12, adjust=False).mean()
        ema26=cached_data['Close'].ewm(span=26, adjust=False).mean()
        
        stock_data['flips'] = process_flips(ema12[-num_days:], ema26[-num_days:])

        #earnings stuffs
        earnings_dates, earnings_diff = get_earnings_history(stock_symbol)
        all_dates = []

        # Calculate dates before the extracted date
        days_before = 3
        for i in range(days_before):
            day = end_date - timedelta(days=i+1)
            all_dates.append(day.strftime("%Y-%m-%d"))

        stock_data['earnings diff'] = []
        for date in all_dates:
            if not end_date in earnings_dates:
                stock_data['earnings diff'].append(0)
                continue
            i = earnings_dates.index(date)
            stock_data['earnings diff'].append(earnings_diff[i])

        # Scale them 0-1
        excluded_values = ['Date', 'earnings diff', 'earnings dates']  # Add any additional columns you want to exclude from scaling
        # Scale each column manually
        for column in information_keys:
            if column in excluded_values:
                continue
            low, high = min(self.data[column]), max(self.data[column])
            column_values = stock_data[column]
            scaled_values = (column_values - low) / (high - low)
            stock_data[column] = scaled_values
        #NOTE: 'Dates' and 'earnings dates' will never be in information_keys
        return [stock_data[key].values.tolist() for key in information_keys]

    def predict(self, info: Optional[np.array] = None) -> np.array:
        """
        This method will wrap the models predict method. It gives the predictions based on data
        from the last 60 days back.  Get the last one to get the applicable day.

        Args: 
            info Optional[np.array]: a np.array of each day. Is optional as if not specified then
            the code will retrieve it nonetheless.
        
        Returns:
            np.array: the predictions for the last 60 days back
        """
        if not info:
            info = self.getInfoToday()
        if self.model:
            #x_train, y_train = create_sequences(info, 60)
            return self.model.predict(info)
        raise LookupError("Compile or load model first")



class DayTradeModel(BaseModel):
    """
    This is the DayTrade child class that inherits from the BaseModel parent class.
    
    It contains the information keys `close`
    """
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
    """
    This is the MACD child class that inherits from the BaseModel parent class.

    It contains the information keys `close`, `MACD`, `Signal Line`, `Histogram`, `flips`, `200-day EMA`
    """
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
    """
    This is the ImpulseMACD child class that inherits from the BaseModel parent class.

    It contains the information keys `close`, `MACD`, `Signal Line`, `Histogram`, `flips`, `200-day EMA`
    """
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
    """
    This is the Reversal child class that inherits from the BaseModel parent class.

    It contains the information keys `close`, `gradual-liquidity spike`, `3-liquidity spike`, `momentum_oscillator`
    """
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
    """
    This is the Earnings child class that inherits from the BaseModel parent class.

    It contains the information keys `close`, `earnings dates`, `earnings diff`, `Momentum`
    """
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close', 'earnings dates', 'earnings diff', 'Momentum']
        )


class BreakoutModel(BaseModel):
    """
    This is the Breakout child class that inherits from the BaseModel parent class.

    It contains the information keys `close`, `RSI`, `TRAMA`
    """
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close', 'RSI', 'TRAMA']
        )


if __name__ == "__main__":
    modelclasses = [ImpulseMACDModel]#[DayTradeModel, MACDModel, ImpulseMACDModel, ReversalModel, EarningsModel, BreakoutModel]
    models = []
    for modelclass in modelclasses:
        model = modelclass()
        model.train()
        models.append(model)
    for model in models:
        print(type(model))
        model.test()
