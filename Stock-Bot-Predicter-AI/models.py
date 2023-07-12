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

from typing import Optional, List, Dict, Union, Any
from warnings import warn
from datetime import datetime, timedelta

from typing_extensions import Self
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from pandas_market_calendars import get_calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from trading_funcs import (
    check_for_holidays, get_relavant_values,
    create_sequences, process_flips,
    excluded_values, is_floats,
    company_symbols
)
from get_info import (
    calculate_momentum_oscillator,
    get_liquidity_spikes,
    get_earnings_history
)


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

    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-07-09",
                 stock_symbol: str = "AAPL",
                 num_days: int = 60,
                 information_keys: List[str]=["Close"]) -> None:
        self.start_date, self.end_date = check_for_holidays(
            start_date, end_date
        )
        self.start_date, self.end_date = check_for_holidays(
            start_date, end_date
        )

        self.stock_symbol = stock_symbol
        self.information_keys = information_keys
        self.num_days = num_days

        self.model: Optional[Sequential] = None
        self.data: Optional[Dict[str, Any]] = None
        self.scaler_data: Dict[str, float] = {}

#________For offline predicting____________#
        self.cached: Optional[np.ndarray] = None

        # NOTE: cached_info is a pd.DateFrame online,
        # while it is a Dict offline
        self.cached_info: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None

    def train(self, epochs: int=100) -> None:
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
        self.data, data, self.scaler_data, start_date, end_date = get_relavant_values(
            start_date, end_date, stock_symbol, information_keys, None
        )
        shape = data.shape[1]

        #_________________Process Data for LSTM______________________#
        # Split the data into training and testing sets
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

        x_total, y_total = create_sequences(train_data, num_days)
        x_train, y_train = create_sequences(train_data, num_days)
        x_test, y_test = create_sequences(test_data, num_days)

        if len(test_data) < num_days:
            raise ValueError('The length of test_data must be more then num days \n increase the data or decrease the num days')
        if len(test_data) < num_days:
            raise ValueError('The length of test_data must be more then num days \n increase the data or decrease the num days')

        #_________________Train it______________________#
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(num_days, shape)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')


        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        # Train the model
        model.fit(x_test, y_test, validation_data=(x_test, y_test), callbacks=[early_stopping], batch_size=32, epochs=epochs)
        model.fit(x_train, y_train, validation_data=(x_train, y_train), callbacks=[early_stopping], batch_size=32, epochs=epochs)
        model.fit(x_total, y_total, validation_data=(x_total, y_total), callbacks=[early_stopping], batch_size=32, epochs=epochs)

        self.model = model

    def save(self) -> None:
        """
        This method will save the model using the tensorflow save method. It will also save the data
        into the `json` file format.
        """
        if self.model is None:
            raise LookupError("Compile or load model first")
        name = self.__class__.__name__

        #_________________Save Model______________________#
        self.model.save(f"Stocks/{self.stock_symbol}/{name}_model")

        for key, val in self.data.items():
            print(key)
            print(type(val))
        with open(f"Stocks/{self.stock_symbol}/{name}_data.json", "w") as json_file:
            json.dump(self.data, json_file)

        with open(f"Stocks/{self.stock_symbol}/{name}_min_max_data.json", "w") as json_file:
            json.dump(self.scaler_data, json_file)

    def is_homogeneous(self, arr) -> bool:
        return len(set(arr.dtype for arr in arr.flatten())) == 1

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
        _, data, _, start_date, end_date = get_relavant_values( # type: ignore[arg-type]
            start_date, end_date, stock_symbol, information_keys, self.scaler_data
        )

        #_________________Process Data for LSTM______________________#
        # Split the data into training and testing sets
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size] # First `num_days` not in predictions
        test_data = data[train_size-num_days-1:] # minus by `num_days` to get full range of values during the test period 

        x_train, y_train = create_sequences(train_data, num_days)
        x_test, y_test = create_sequences(test_data, num_days)
        #_________________TEST QUALITY______________________#
        train_predictions = self.model.predict(x_train)
        test_predictions = self.model.predict(x_test)

        # NOTE: This cuts data at the start to account for `num_days`
        train_data = data[num_days:train_size]
        test_data = data[train_size-1:]

        assert len(train_predictions) == len(train_data)
        assert len(test_predictions) == len(test_data)


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
    
        print("Homogeneous(Should be True):")
        assert self.is_homogeneous(data)

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
            [predicted_test[0], actual_test[0], predicted_train[0], actual_train[0]],#[real_data, actual_test[0], actual_train],
            ['Predicted Test', 'Actual Test', 'Predicted Train', 'Actual Train']#['Real Data', 'Actual Test', 'Actual Train']
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
            return None
        name = self.__class__.__name__

        self.model = load_model(f"Stocks/{self.stock_symbol}/{name}_model")

        with open(f"Stocks/{self.stock_symbol}/{name}_data.json", 'r') as file:
            self.data = json.load(file)

        with open(f"Stocks/{self.stock_symbol}/{name}_min_max_data.json", 'r') as file:
            self.scaler_data = json.load(file)

        # type: ignore[no-any-return]
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

        stock_data['Close'] = cached_info['Close'].iloc[-num_days:]

        ema12 = cached_info['Close'].ewm(span=12, adjust=False).mean()
        ema26 = cached_info['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        span = 9
        signal_line = macd.rolling(window=span, min_periods=1).mean().iloc[-num_days:]

        change = cached_info['Close'].diff()
        if '12-day EMA' in information_keys:
            stock_data['12-day EMA'] = ema12.iloc[-num_days:]
        if '26-day EMA' in information_keys:
            stock_data['26-day EMA'] = ema26.iloc[-num_days:]
        if 'MACD' in information_keys:
            stock_data['MACD'] = macd.iloc[-num_days:]
        if 'Signal Line' in information_keys:
            stock_data['Signal Line'] = signal_line
        if 'Histogram' in information_keys:
            histogram = macd - signal_line
            stock_data['Histogram'] = histogram.iloc[-num_days:]
        if '200-day EMA' in information_keys:
            ewm200 = cached_info['Close'].ewm(span=200, adjust=False)
            ema200 = ewm200.mean().iloc[-num_days:]
            stock_data['200-day EMA'] = ema200
        change = cached_info['Close'].diff().iloc[-num_days:]
        if 'Change' in information_keys:
            stock_data['Change'] = change.iloc[-num_days:]
        if 'Momentum' in information_keys:
            momentum = change.rolling(window=10, min_periods=1).sum().iloc[-num_days:]
            stock_data['Momentum'] = momentum
        if 'RSI' in information_keys:
            gain = change.apply(lambda x: x if x > 0 else 0)
            loss = change.apply(lambda x: abs(x) if x < 0 else 0)
            avg_gain = gain.rolling(window=14).mean().iloc[-num_days:]
            avg_loss = loss.rolling(window=14).mean().iloc[-num_days:]
            relative_strength = avg_gain / avg_loss
            stock_data['RSI'] = 100 - (100 / (1 + relative_strength))
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
        if '3-liquidity spike' in information_keys:
            stock_data['3-liquidity spike'] = get_liquidity_spikes(
                cached_info['Volume'], z_score_threshold=4
            ).iloc[-num_days:]
        if 'momentum_oscillator' in information_keys:
            stock_data['momentum_oscillator'] = calculate_momentum_oscillator(
                cached_info['Close']
            ).iloc[-num_days:]
        if 'ema_flips' in information_keys:
            #_________________12 and 26 day Ema flips______________________#
            stock_data['ema_flips'] = process_flips(ema12[-num_days:], ema26[-num_days:])
            stock_data['ema_flips'] = pd.Series(stock_data['ema_flips'])
        if 'signal_flips' in information_keys:
            stock_data['signal_flips'] = process_flips(macd[-num_days:], signal_line[-num_days:])
            stock_data['signal_flips'] = pd.Series(stock_data['signal_flips'])
        if 'earning diffs' in information_keys:
            #earnings stuffs
            earnings_dates, earnings_diff = get_earnings_history(self.stock_symbol)
            
            end_datetime = datetime.strptime(self.end_date, "%Y-%m-%d")
            date = end_datetime - timedelta(days=num_days)

            stock_data['earnings dates'] = []
            stock_data['earning diffs'] = [] # type: ignore[attr]
            low = self.scaler_data['earning diffs']['min'] # type: ignore[index]
            diff = self.scaler_data['earning diffs']['diff'] # type: ignore[index]

            for i in range(num_days):
                if not self.end_date in earnings_dates:
                    stock_data['earning diffs'].append(0)
                    continue
                i = earnings_dates.index(date)
                scaled = (earnings_diff[i]-low) / diff
                stock_data['earning diffs'].append(scaled)

        # Scale each column manually
        for column in self.information_keys:
            if column in excluded_values:
                continue
            low = self.scaler_data[column]['min'] # type: ignore[index]
            diff = self.scaler_data[column]['diff'] # type: ignore[index]
            diff = self.scaler_data[column]['diff'] # type: ignore[index]
            column_values = stock_data[column]
            scaled_values = (column_values - low) / diff
            scaled_values = (column_values - low) / diff
            stock_data[column] = scaled_values
        return stock_data

    def update_cached_online(self) -> bool:
        """
        This method updates the cached data using the internet.
        
        Returns:
            False: if it is a holiday
            True: if it is a working day
        """
        cached_info = self.cached_info
        cached = self.cached
        num_days = self.num_days
        end_datetime = datetime.strptime(self.end_date, "%Y-%m-%d")

        #_________________ GET Data______________________#
        ticker = yf.Ticker(self.stock_symbol)
        
        #NOTE: optimize bettween
        if self.cached_info is None:
            start_datetime = end_datetime - timedelta(days=280)
            cached_info = ticker.history(start=start_datetime, end=self.end_date, interval="1d")
            if len(cached_info) == 0: # type: ignore[arg-type]
                raise ConnectionError("Stock data failed to load. Check your internet")
        else:
            start_datetime = end_datetime - timedelta(days=1)
            day_info = ticker.history(start=start_datetime, end=self.end_date, interval="1d")
            if len(day_info) == 0: # type: ignore[arg-type]
                raise ConnectionError("Stock data failed to load. Check your internet")
            cached_info = cached_info.drop(cached_info.index[0])
            cached_info = pd.concat((cached_info, day_info))


        cached = self.indicators_past_num_days(cached_info, num_days)
        cached = [cached[key] for key in self.information_keys if is_floats(cached[key])]
        cached = np.transpose(cached)

        self.cached_info = cached_info
        self.cached = cached
        return True # Working Day

    def update_cached_offline(self) -> None:
        """This method updates the cached data without using the internet."""
        warn("For Testing")

        end_date = self.end_date
        #_________________ GET Data______________________#
        if not self.cached_info:
            with open(f"Stocks/{self.stock_symbol}/info.json", 'r') as file:
                cached_info = json.load(file)

                if not self.start_date in cached_info['Dates']:
                    raise ValueError("start is before or after `Dates` range")
                if not self.end_date in cached_info['Dates']:
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

    def get_info_today(self) -> Optional[np.ndarray]:
        """
        This method will get the information for the stock today and the
        last relevant days to the stock.

        The cached_data is used so less data has to be retrieved from
        yf.finance as it is held to cached or something else.
        
        Returns:
            np.array: The information for the stock today and the
                last relevant days to the stock
        """
        end_datetime = datetime.strptime(self.end_date, "%Y-%m-%d")

        start_datetime = end_datetime - timedelta(days=1)
        nyse = get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start_datetime, end_date=end_datetime+timedelta(days=2))
        if self.end_date not in schedule.index:
            return None

        try:
            if type(self.cached_info) is Dict:
                raise ConnectionError("It has already failed to lead")
            self.update_cached_online()
        except ConnectionError:
            warn("Stock data failed to download. Check your internet")
            if type(self.cached_info) is pd.DataFrame:
                self.cached_info = None
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
        >>> obj = BaseModel(num_days=5)
        >>> obj = BaseModel(num_days=5)
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
        if info is None: # basically, if it is still None after get_info_today
            raise RuntimeError(
                "Could not get indicators for today. It may be that `end_date` is beyond today's date"
            )
        if self.model:
            return self.model.predict(info) # typing: ignore[return]
        raise LookupError("Compile or load model first")


class DayTradeModel(BaseModel):
    """
    This is the DayTrade child class that inherits from
    the BaseModel parent class.
    
    It contains the information keys `Close`
    """
    def __init__(self,
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            stock_symbol=stock_symbol,
            information_keys=['Close']
        )


class MACDModel(BaseModel):
    """
    This is the MACD child class that inherits
    from the BaseModel parent class.

    It contains the information keys `Close`, `MACD`,
    `Signal Line`, `Histogram`, `ema_flips`, `200-day EMA`
    """
    def __init__(self,
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            stock_symbol=stock_symbol,
            information_keys=['Close', 'MACD', 'Histogram', 'ema_flips', '200-day EMA']
        )


class ImpulseMACDModel(BaseModel):
    """
    This is the ImpulseMACD child class that inherits from
    the BaseModel parent class.

    The difference between this class and the MACD model class is that the Impluse MACD model
    is more responsive to short-term market changes and can identify trends earlier. 

    It contains the information keys `Close`, `Histogram`, `Momentum`,
    `Change`, `Histogram`, `ema_flips`, `signal_flips`, `200-day EMA`
    """
    def __init__(self,
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            stock_symbol=stock_symbol,
            information_keys=['Close', 'Histogram', 'Momentum', 'Change', 'ema_flips', 'signal_flips', '200-day EMA']
        )


class ReversalModel(BaseModel):
    """
    This is the Reversal child class that inherits from
    the BaseModel parent class.

    It contains the information keys `Close`, `gradual-liquidity spike`,
    `3-liquidity spike`, `momentum_oscillator`
    """
    def __init__(self,
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
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
    `earning diffs`, `Momentum`
    """
    def __init__(self,
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            stock_symbol=stock_symbol,
            information_keys=['Close', 'earnings dates', 'earning diffs', 'Momentum']
        )


class BreakoutModel(BaseModel):
    """
    This is the Breakout child class that inherits from
    the BaseModel parent class.

    It contains the information keys `Close`, `RSI`, `TRAMA`
    """
    def __init__(self,
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
            stock_symbol=stock_symbol,
            information_keys=['Close', 'RSI', 'TRAMA']
        )


class RSIModel(BaseModel):
    """
    This is the Breakout child class that inherits from
    the BaseModel parent class.

    It contains the information keys `Close`, `RSI`, `TRAMA`, `Bollinger Middle`,
                `Above Bollinger`, `Bellow Bollinger`, `Momentum`
    """
    def __init__(self,
                 stock_symbol: str = "AAPL") -> None:
        super().__init__(
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
    def __init__(self,
                 stock_symbol: str = "AAPL") -> None:
        raise Warning("BUGGED, NOT WORKING")
        super().__init__(
            stock_symbol=stock_symbol,
            information_keys=[
                'Close', 'supertrend1', 'supertrend2',
                'supertrend3', '200-day EMA', 'kumo_cloud'
            ]
        )


if __name__ == "__main__":
    modelclasses = [ImpulseMACDModel]#, EarningsModel, RSIModel]

    test_models = []
    for company in company_symbols:
        for modelclass in modelclasses:
            model = modelclass(stock_symbol=company)
            model.train(epochs=100)
            model.save()
            #test_models.append(model)

        #for model in test_models:
        #    model.test()
