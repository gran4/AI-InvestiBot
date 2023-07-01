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

from typing import Optional, List, Dict, Union
from warnings import warn
from datetime import datetime, timedelta

from typing_extensions import Self
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from trading_funcs import check_for_holidays, get_relavant_values, create_sequences, process_flips, excluded_values
from get_info import calculate_momentum_oscillator, get_liquidity_spikes, get_earnings_history


__all__ = (
    'BaseModel',
    'DayTradeModel',
    'MACDModel',
    'ImpulseMACDModel',
    'ReversalModel',
    'EarningsModel',
    'BreakoutModel',
    'RSIModel',
    'SuperTrendsModel'
)


class BaseModel:
    """
    This is the base class for all the models. It handles the actual training, saving,
    loading, predicting, etc. Setting the `information_keys` allows us to describe what
    the model uses. The information keys themselves are retrieved from a json format
    that was created by getInfo.py.

    Args:
        start_date (str): The start date of the training data
        end_date (str): The end date of the training data
        stock_symbol (str): The stock symbol of the stock you want to train on
        num_days (int): The number of days to use for the LSTM model
        information_keys (List[str]): The information keys that describe what the model uses
    """

    def __init__(self, start_date: str = "2021-01-01",
                 end_date: str = "2022-01-05",
                 stock_symbol: str = "AAPL",
                 num_days: int = 60,
                 information_keys: List[str]=["Close"]) -> None:
        self.set_dates(start_date, end_date)

        self.stock_symbol = stock_symbol
        self.information_keys = information_keys
        self.num_days = num_days

        self.model: Optional[Sequential] = None
        self.data: Optional[Dict] = None
        self.scaler_data: Optional[Dict] = None

#________For offline predicting____________#
        self.cached: Optional[np.ndarray] = None

        # NOTE: cached_info is a pd.DateFrame online,
        # while it is a Dict offline
        self.cached_info: Optional[Union[pd.DataFrame, Dict]] = None

    def train(self, epochs: int=10) -> None:
        """
        Trains Model off `information_keys`

        Args:
            epochs (int): The number of epochs to train the model for
        """
        warn("If you saved before, use load func instead")

        start_date = self.start_date
        end_date = self.end_date
        stock_symbol = self.stock_symbol
        information_keys = self.information_keys
        num_days = self.num_days

        #_________________ GET Data______________________#
        self.data, data, start_date, end_date = get_relavant_values(
            start_date, end_date, stock_symbol, information_keys
        )
        shape = data.shape[1]

        for key in self.data.keys():
            self.data[key] = list(self.data[key])

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

        #For predictions
        self.scaler_data = {}
        for key, val in self.data.items():
            self.scaler_data[key] = {"min": min(val), "max": max(val)}

    def save(self) -> None:
        """
        This method will save the model using the tensorflow save method. It will also save the data
        into the `json` file format.
        """
        if self.model is None:
            raise LookupError("Compile or load model first")
        #_________________Save Model______________________#
        self.model.save(f"{self.stock_symbol}/model")

        with open(f"{self.stock_symbol}/data.json", "w") as json_file:
            json.dump(self.data, json_file)

        with open(f"{self.stock_symbol}/min_max_data.json", "w") as json_file:
            json.dump(self.scaler_data, json_file)

    def test(self) -> None:
        """
        A method for testing purposes. 
        
        Warning:
            It is EXPENSIVE.
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
        _, data, start_date, end_date = get_relavant_values(
            start_date, end_date, stock_symbol, information_keys
        )

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
        def is_homogeneous(arr) -> bool:
            return len(set(arr.dtype for arr in arr.flatten())) == 1
        print("Homogeneous(Should be True):", is_homogeneous(data))

        y_train_size = y_train.shape[0]
        days_test = list(range(y_train.shape[0]))
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
        plt.legend(
            [predicted_test[0], actual_test[0], predicted_train[0], actual_train[0]],
            ['Predicted Test', 'Actual Test', 'Predicted Train', 'Actual Train']
        )
        plt.show()

    def load(self) -> Optional[Self]:
        """
        This method will load the model using the tensorflow load method.

        Returns:
            None: If no model is loaded
            BaseModel: The saved model if it was successfully saved
        """
        if self.model:
            return
        self.model = load_model(f"{self.stock_symbol}/model")

        with open(f"{self.stock_symbol}/data.json", 'r') as file:
            self.data = json.load(file)

        with open(f"{self.stock_symbol}/min_max_data.json", 'r') as file:
            self.scaler_data = json.load(file)

        return self.model

    def indicators_past_num_days(self, cached_info: pd.DataFrame,
                                 num_days: int) -> Dict[str, Union[float, str]]:
        """
        This method will return the indicators for the past `num_days` days specified in the
        information keys. It will use the cached information to calculate the indicators
        until the `end_date`.

        Args:
            cached_info (pd.DataFrame): The cached information
            num_days (int): The number of days to calculate the indicators for
        
        Returns:
            dict: A dictionary containing the indicators for the stock data
                Values will be floats except some expections tht need to be
                processed during run time
        """
        stock_data = {}
        information_keys = self.information_keys

        if '12-day EMA' in information_keys:
            ewm12 = cached_info['Close'].ewm(span=12, adjust=False)
            ema12 = ewm12.mean().iloc[-num_days:]
            stock_data['12-day EMA'] = ema12
        if '26-day EMA' in information_keys:
            ewm26 = cached_info['Close'].ewm(span=26, adjust=False)
            ema26 = ewm26.mean().iloc[-num_days:]
            stock_data['26-day EMA'] = ema26
        if 'MACD' in information_keys:
            macd = ema12 - ema26
            stock_data['MACD'] = macd
        if 'Signal Line' in information_keys:
            span = 9
            signal_line = macd.rolling(window=span).mean().iloc[-num_days:]
            stock_data['Signal Line'] = signal_line
        if 'Histogram' in information_keys:
            histogram = macd - stock_data['Signal Line']
            stock_data['Histogram'] = histogram
        if '200-day EMA' in information_keys:
            ewm200 = cached_info['Close'].ewm(span=200, adjust=False)
            ema200 = ewm200.mean().iloc[-num_days:]
            stock_data['200-day EMA'] = ema200
        if 'Change' in information_keys:
            change = cached_info['Close'].diff().iloc[-num_days:]
            stock_data['Change'] = change
        if 'Momentum' in information_keys:
            momentum = change.rolling(window=10, min_periods=1).sum().iloc[-num_days:]
            stock_data['Momentum'] = momentum
        if 'RSI' in information_keys:
            gain = change.apply(lambda x: x if x > 0 else 0)
            loss = change.apply(lambda x: abs(x) if x < 0 else 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean().iloc[-num_days:]
            avg_loss = loss.rolling(window=14, min_periods=1).mean().iloc[-num_days:]
            rs = avg_gain / avg_loss
            stock_data['RSI'] = 100 - (100 / (1 + rs))
        if 'TRAMA' in information_keys:
            # TRAMA
            volatility = cached_info['Close'].diff().abs().iloc[-num_days:]
            trama = cached_info['Close'].rolling(window=14).mean().iloc[-num_days:]
            stock_data['TRAMA'] = trama + (volatility * 0.1)
        if 'gradual-liquidity spike' in information_keys:
            # Reversal
            stock_data['gradual-liquidity spike'] = get_liquidity_spikes(
                cached_info['Volume'], gradual=True
            ).iloc[-num_days:]
            stock_data['3-liquidity spike'] = get_liquidity_spikes(
                cached_info['Volume'], z_score_threshold=4
            ).iloc[-num_days:]
            stock_data['momentum_oscillator'] = calculate_momentum_oscillator(
                cached_info['Close']).iloc[-num_days:]
        if 'flips' in information_keys:
            #_________________12 and 26 day Ema flips______________________#
            ema12=cached_info['Close'].ewm(span=12, adjust=False).mean()
            ema26=cached_info['Close'].ewm(span=26, adjust=False).mean()

            stock_data['flips'] = process_flips(ema12[-num_days:], ema26[-num_days:])
        if 'earnings diff' in information_keys:
            #earnings stuffs
            earnings_dates, earnings_diff = get_earnings_history(self.stock_symbol)
            all_dates = []

            # Calculate dates before the extracted date
            days_before = 3
            end_datetime = datetime.strptime(self.end_date, "%Y-%m-%d")
            for i in range(days_before):
                day = end_datetime - timedelta(days=i+1)
                all_dates.append(day.strftime('%Y-%m-%d'))

            stock_data['earnings diff'] = []
            low = self.scaler_data['earnings diffs']['min']
            high = self.scaler_data['earnings diffs']['max']
            for date in all_dates:
                if not self.end_date in earnings_dates:
                    stock_data['earnings diff'].append(0)
                    continue
                i = earnings_dates.index(date)
                scaled = (earnings_diff[i]-low) / (high - low)
                stock_data['earnings diff'].append(scaled)

        # Scale each column manually
        for column in self.information_keys:
            if column in excluded_values:
                continue
            low = self.scaler_data[column]['min']
            high = self.scaler_data[column]['max']
            column_values = stock_data[column]
            scaled_values = (column_values - low) / (high - low)
            stock_data[column] = scaled_values
        return stock_data

    def indicators_today(self, day_info: pd.DataFrame,
                         end_date: datetime, num_days: int
                         ) -> Dict[str, Union[float, str]]:
        """
        This method calculates the indicators for the stock data for the current day. 
        It will use the current day_info to calculate the indicators until the `end_date`.

        Args:
            day_info (pd.DataFrame): The stock data for the current day
            end_date (datetime): The date to stop calculating the indicators
            num_days (int): The number of days to calculate the indicators for
        
        Returns: 
            dict: A dictionary of the indicators for the stock data
                Values will be floats except some expections tht need to be processed during run time
        """
        raise NotImplementedError("Will be added soon")

    def update_cached_online(self) -> None:
        """This method updates the cached data using the internet."""
        cached_info = self.cached_info
        cached = self.cached
        num_days = self.num_days
        end_datetime = datetime.strptime(self.end_date, "%Y-%m-%d")

        #_________________ GET Data______________________#
        ticker = yf.Ticker(self.stock_symbol)
        if self.cached_info: #is more than 0
            # Only get one day's info
            day_data = ticker.history(start=self.end_date, end=self.end_date, interval="1d")

            if not len(day_data):
                raise ConnectionError("Stock data failed to load. Check your internet")

            for key, val in day_data.items():
                cached_info[key].pop()######## NOTE NOTE NOTE
                cached_info[key].append(val)# NOTE DO IT FINISH

            day_data = self.indicators_today(cached_info, end_datetime, num_days)

            # make sure it is in correct order
            day_data = [day_data[key] for key in self.information_keys]
            #delete first day and add new day.
            cached = np.concatenate((cached[1:], [day_data]))
        else:
            start_date = end_datetime - timedelta(days=260)
            cached_info = ticker.history(start=start_date, end=self.end_date, interval="1d")

            if not len(cached_info):
                raise ConnectionError("Stock data failed to load. Check your internet")
            cached = self.indicators_past_num_days(cached_info, num_days)

            cached = [cached[key] for key in self.information_keys]
            cached = np.transpose(cached)
        self.cached_info = cached_info
        self.cached = cached

    def set_dates(self, start_date: str, end_date: str) -> None:
        """Wrapper to setting dates that checks for holidays"""
        self.start_date, self.end_date = check_for_holidays(
            start_date, end_date
        )

    def update_cached_offline(self) -> None:
        """This method updates the cached data without using the internet."""
        warn("For Testing")

        end_date = self.end_date
        #_________________ GET Data______________________#
        if not self.cached_info:
            with open(f"{self.stock_symbol}/info.json", 'r') as file:
                cached_info = json.load(file)

                if not self.start_date in cached_info['Dates']:
                    raise ValueError("start is before or after `Dates` range")
                elif not self.end_date in cached_info['Dates']:
                    raise ValueError("end is before or after `Dates` range")

                end_index = cached_info["Dates"].index(self.end_date)
                cached = []
                for key in self.information_keys:
                    if key not in excluded_values:
                        cached.append(
                            cached_info[key][end_index-self.num_days:end_index]
                        )
                cached = np.transpose(cached)

                self.cached = cached
                self.cached_info = cached_info
            if len(self.cached) == 0:
                raise RuntimeError("Stock data failed to load. Reason Unknown")
        if len(self.cached) != 0:
            i_end = self.cached_info["Dates"].index(end_date)
            day_data = [self.cached_info[key][i_end] for key in self.information_keys]

            #delete first day and add new day.
            self.cached = np.concatenate((self.cached[1:], [day_data]))

    def get_info_today(self) -> np.ndarray:
        """
        This method will get the information for the stock today and the
        last relevant days to the stock.

        The cached_data is used so less data has to be retrieved from
        yf.finance as it is held to cached or something else.
        
        Returns:
            np.array: The information for the stock today and the
                last relevant days to the stock
        """
        try:
            self.update_cached_online()
        except ConnectionError:
            warn("Stock data failed to download. Check your internet")
            self.update_cached_offline()

        if self.cached is None:
            raise RuntimeError('Neither the online or offline updating of `cached` worked')

        date_object = datetime.strptime(self.start_date, "%Y-%m-%d")
        next_day = date_object + timedelta(days=1)
        self.start_date = next_day.strftime("%Y-%m-%d")

        date_object = datetime.strptime(self.end_date, "%Y-%m-%d")
        next_day = date_object + timedelta(days=1)
        self.end_date = next_day.strftime("%Y-%m-%d")

        #NOTE: 'Dates' and 'earnings dates' will never be in information_keys
        self.cached = np.reshape(self.cached, (1, 60, self.cached.shape[1]))
        return self.cached

    def predict(self, info: Optional[np.ndarray] = None) -> np.ndarray:
        """
        This method wraps the model's predict method using `info`.

        Args: 
            info (Optional[np.ndarray]): the information to predict on.
            If None, it will get the info from the last relevant days back.
        
        Returns:
            np.ndarray: the predictions of the model
                The length is determined by how many are put in.
                So, you can predict for time frames or one day
                depending on what you want.
                The length is the days `info` minus `num_days` plus 1

        :Example:
        >>> obj = BaseModel(42)
        >>> obj.num_days
        5
        >>> temp = obj.predict(info = np.array(
                [2, 2],
                [3, 2],
                [4, 1],
                [3, 2],
                [0, 2]
                [7, 0],
                [1, 2],
                [0, 1],
                [2, 2],
                )
            ))
        >>> print(len(temp))
        4
        """
        if info is None:
            info = self.get_info_today()
        if self.model:
            return self.model.predict(info)
        raise LookupError("Compile or load model first")



class DayTradeModel(BaseModel):
    """
    This is the DayTrade child class that inherits from
    the BaseModel parent class.
    
    It contains the information keys `Close`
    """
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-02-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close']
        )


class MACDModel(BaseModel):
    """
    This is the MACD child class that inherits
    from the BaseModel parent class.

    It contains the information keys `Close`, `MACD`,
    `Signal Line`, `Histogram`, `flips`, `200-day EMA`
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
    This is the ImpulseMACD child class that inherits from
    the BaseModel parent class.

    The difference between this class and the MACD model class is that the Impluse MACD model
    is more responsive to short-term market changes and can identify trends earlier. 

    It contains the information keys `Close`, `MACD`,
    `Signal Line`, `Histogram`, `flips`, `200-day EMA`
    """
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2020-07-06",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=['Close', 'Histogram', 'Momentum', 'Change', 'flips', '200-day EMA']
        )


class ReversalModel(BaseModel):
    """
    This is the Reversal child class that inherits from
    the BaseModel parent class.

    It contains the information keys `Close`, `gradual-liquidity spike`,
    `3-liquidity spike`, `momentum_oscillator`
    """
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=[
                'Close', 'gradual-liquidity spike',
                '3-liquidity spike', 'momentum_oscillator'
            ]
        )


class EarningsModel(BaseModel):
    """
    This is the Earnings child class that inherits from
    the BaseModel parent class.

    It contains the information keys `Close`, `earnings dates`,
    `earnings diff`, `Momentum`
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
    This is the Breakout child class that inherits from
    the BaseModel parent class.

    It contains the information keys `Close`, `RSI`, `TRAMA`
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


class RSIModel(BaseModel):
    """
    This is the Breakout child class that inherits from
    the BaseModel parent class.

    It contains the information keys `Close`, `RSI`, `TRAMA`
    """
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=[
                'Close', 'RSI', 'TRAMA', 'Bollinger Middle',
                'Above Bollinger', 'Bellow Bollinger'
            ]
        )


class RSIModel2(BaseModel):
    """
    This is the Breakout child class that inherits from
    the BaseModel parent class.

    It contains the information keys `Close`, `RSI`, `TRAMA`
    """
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=[
                'Close', 'RSI', 'TRAMA', 'Bollinger Middle',
                'Above Bollinger', 'Bellow Bollinger', 'Momentum'
            ]
        )


class SuperTrendsModel(BaseModel):
    """
    This is the Breakout child class that inherits from
    the BaseModel parent class.

    It contains the information keys `Close`, `RSI`, `TRAMA`
    """
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            stock_symbol=stock_symbol,
            information_keys=[
                'Close', 'supertrend1', 'supertrend2',
                'supertrend3', '200-day EMA', 'kumo_cloud'
            ]
        )


if __name__ == "__main__":
    #[DayTradeModel, MACDModel, ImpulseMACDModel, ReversalModel, EarningsModel, BreakoutModel]
    modelclasses = [DayTradeModel]

    test_models = []
    for modelclass in modelclasses:
        model = modelclass()
        model.train(epochs=100)
        test_models.append(model)
        model.save()

    for model in test_models:
        model.test()
