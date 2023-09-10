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
import os

from typing import Any, Optional, Union, Callable, List, Dict
from warnings import warn
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from custom_objects import create_LSTM_model, create_LSTM_model2
from custom_objects import *

from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from pandas_market_calendars import get_calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from trading_funcs import (
    check_for_holidays, get_relavant_values,
    create_sequences, process_flips,
    non_daily, non_daily_no_use, is_floats,
    calculate_percentage_movement_together,
    indicators_to_add_noise_to, indicators_to_scale,
)
from get_info import (
    calculate_momentum_oscillator,
    get_liquidity_spikes,
    get_earnings_history
)


__all__ = (
    'CustomLoss',
    'CustomLoss2',
    'BaseModel',
    'PriceModel',
    'PercentageModel',
    'ImpulseMACD_indicators',
    'Reversal_indicators',
    'RSI_indicators',
    'Earnings_indicators',
    'break_out_indicators', 
    'super_trends_indicators'
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

    def __init__(self, start_date: str = None,
                 end_date: Optional[Union[date, str]] = None,
                 stock_symbol: Optional[Union[date, str]] = "AAPL",
                 num_days: int = None,
                 information_keys: List[str]=["Close"]) -> None:
        if num_days is None:
            with open(f'Stocks/{stock_symbol}/dynamic_tuning.json', 'r') as file:
                num_days = json.load(file)['num_days']

        self.stock_symbol = stock_symbol
        self.information_keys = information_keys
        self.num_days = num_days

        self.update_dates(start_date=start_date, end_date=end_date)

        self.model: Optional[Sequential] = None
        self.scaler_data: Dict[str, float] = {}

#________For offline predicting____________#
        self.cached: Optional[np.ndarray] = None

        # NOTE: cached_info is a pd.DateFrame online,
        # while it is a Dict offline
        self.cached_info: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None

    def update_dates(
            self, start_date=None,
            end_date=None,
        ):
        if end_date is None:
            end_date = date.today()
            #lower type(end_date) == date turns it into string
        if start_date is None:
            with open(f'Stocks/{self.stock_symbol}/dynamic_tuning.json', 'r') as file:
                relevant_years = json.load(file)['relevant_years']
            start_date = end_date - relativedelta(years=relevant_years)

        if type(end_date) == date:
            end_date = end_date.strftime("%Y-%m-%d")
        if type(start_date) == date:
            start_date = start_date.strftime("%Y-%m-%d")
        self.start_date, self.end_date = check_for_holidays(
            start_date, end_date
        )

    def process_x_y_total(self, x_total, y_total, num_days, time_shift):
        return x_total, y_total

    def train(self, epochs: int=1000,
              patience: int=5, time_shift: int=0,
              add_scaling: bool=False, add_noise: bool=True,
              use_transfer_learning: bool=False, test: bool=False,
              create_model: Callable=create_LSTM_model) -> None:
        """
        Trains Model off `information_keys`

        Args:
            epochs (int): The number of epochs to train the model for
            patience (int): The amount of epochs of stagnation before early stopping
            time_shift (int): The amount of time to shift the data by(in days)
                EX. allows bot to predict 1 month into the future
            add_scaling (bool): Data Augmentation using scaling
            add_noise (bool): Data Augmentation using noise
            use_transfer_learning (bool): Whether or not to use tranfer learnign model
            test (bool): Whether or not to use the test data
        
        """
        warn("If you saved before, use load func instead")
        if time_shift < 0:
            raise ValueError("`time_shift` must be equal of greater than 0")

        start_date = self.start_date
        end_date = self.end_date
        stock_symbol = self.stock_symbol
        information_keys = self.information_keys
        num_days = self.num_days
        print(start_date, end_date)

        #_________________ GET Data______________________#
        _, data, self.scaler_data = get_relavant_values(
            stock_symbol, information_keys, start_date=start_date, end_date=end_date
        )

        #_________________Process Data for LSTM______________________#
        split = int(len(data))
        print(data.shape)
        if test:
            x_total, y_total = create_sequences(data[:int(split*.8)], num_days)
        else:
            x_total, y_total = create_sequences(data, num_days)
        x_total, y_total = self.process_x_y_total(x_total, y_total, num_days, time_shift)
        if time_shift != 0: # 0 removes everything
            x_total = x_total[:-time_shift]
            y_total = y_total[time_shift:]

        processed = [key for key in information_keys if not key in non_daily_no_use]
        if len(x_total.shape) == 3:
            model = create_model((num_days, len(processed)))
        else:
            model = create_model((x_total.shape[1], num_days, len(processed)))

        if use_transfer_learning:
            transfer_model = load_model(f"transfer_learning_model")
            for layer_idx, layer in enumerate(model.layers):
                if layer.name in transfer_model.layers[layer_idx].name:
                    layer.set_weights(transfer_model.layers[layer_idx].get_weights())

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        #_________________Train it______________________#
        divider = int(split/2)
        if add_scaling:
            indices_cache = [information_keys.index(key) for key in indicators_to_scale if key in information_keys]

            x_total_copy = np.copy(x_total)
            y_total_copy = np.copy(y_total)
            model.fit(x_total_copy*1.1, y_total_copy*1.1, validation_data=(x_total_copy, y_total_copy), callbacks=[early_stopping], batch_size=64, epochs=epochs)

            x_total_p1 = np.copy(x_total[:divider])
            y_total_p1 = np.copy(y_total[:divider])
            x_total_p1[:, indices_cache] *= 2
            y_total_p1 *= 2
        if add_noise:
            x_total_copy = np.copy(x_total)
            y_total_copy = np.copy(y_total)
            # Get the indices of indicators to add noise to
            indices_cache = [information_keys.index(key) for key in indicators_to_add_noise_to if key in information_keys]

            # Create a noise array with the same shape as x_total's selected columns
            noise = np.random.uniform(-0.001, 0.001)
            # Add noise to the selected columns of x_total
            x_total_copy[:, indices_cache] += noise
            y_total_copy += np.random.uniform(-0.001, 0.001, size=y_total.shape[0])
            model.fit(x_total, y_total, validation_data=(x_total, y_total), callbacks=[early_stopping], batch_size=64, epochs=epochs)

        #Ties it together on the real data
        model.fit(x_total, y_total, validation_data=(x_total, y_total), callbacks=[early_stopping], batch_size=64, epochs=epochs)
        self.model = model

    def save(self, transfer_learning: bool=False, name: Optional[str]=None) -> None:
        """
        This method will save the model using the tensorflow save method. It will also save the data
        into the `json` file format.
        """
        if self.model is None:
            raise LookupError("Compile or load model first")
        if name is None:
            name = self.__class__.__name__

        #_________________Save Model______________________#
        if transfer_learning:
            self.model.save(f"transfer_learning_model")
            return
        self.model.save(f"Stocks/{self.stock_symbol}/{name}_model")

        if os.path.exists(f'Stocks/{self.stock_symbol}/min_max_data.json'):
            with open(f"Stocks/{self.stock_symbol}/min_max_data.json", 'r') as file:
                temp = json.load(file)
            self.scaler_data.update({key: value for key, value in temp.items()})

        with open(f"Stocks/{self.stock_symbol}/min_max_data.json", "w") as json_file:
            json.dump(self.scaler_data, json_file)

    @staticmethod
    def plot(data):
        """Plots any np.array that you give in"""
        days_train = [i for i in range(data.shape[0])]
        data = data[:, 0]
        # Plot the actual and predicted prices
        plt.figure(figsize=(18, 6))

        predicted_test = plt.plot(days_train, data, label='Predicted Test')
        plt.title(f'TITLE')
        plt.xlabel("X")
        plt.ylabel("Y")

        import matplotlib.ticker as ticker
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(7))

        plt.legend(
            [predicted_test[0]],#[real_data, actual_test[0], actual_train],
            ['Data']#['Real Data', 'Actual Test', 'Actual Train']
        )
        plt.show()

    @staticmethod
    def is_homogeneous(arr) -> bool:
        """Checks if any of the models indicators are missing"""
        return len(set(arr.dtype for arr in arr.flatten())) == 1

    def test(self, time_shift: int=0, show_graph: bool=False,
             title: str="Stock Price Prediction", x_label: str='', y_label: str='Price'
             ) -> None:
        """
        A method for testing purposes. 
        
        Args:
            time_shift (int): The amount of time to shift the data by(in days)
                EX. allows bot to predict 1 month into the future

        Warning:
            It is EXPENSIVE.
        """
        warn("Expensive, for testing purposes")

        if not self.model:
            raise LookupError("Compile or load model first")

        if time_shift < 0:
            raise ValueError("`time_shift` must be equal of greater than 0")

        start_date = self.start_date
        end_date = self.end_date
        stock_symbol = self.stock_symbol
        information_keys = self.information_keys
        num_days = self.num_days

        #_________________ GET Data______________________#
        total_data_dict, data, _ = get_relavant_values( # type: ignore[arg-type]
            stock_symbol, information_keys, self.scaler_data, start_date, end_date
        )

        #_________________Process Data for LSTM______________________#
        split = int(len(data) * 0.8)
        test_data = data[split-num_days-1:] # minus by `num_days` to get full range of values during the test period 

        x_test, y_test = create_sequences(test_data, num_days)
        if time_shift != 0:
            x_test = x_test[:-time_shift]
            y_test = y_test[time_shift:]
        x_test, y_test = self.process_x_y_total(x_test, y_test, num_days, time_shift)

        #_________________TEST QUALITY______________________#
        test_predictions = self.model.predict(x_test)

        # NOTE: This cuts data at the start to account for `num_days`
        if time_shift > 0:
            test_data = data[:-time_shift]
        directional_test, spatial_test = calculate_percentage_movement_together(y_test, test_predictions)
        print("Directional Test: ", directional_test)
        print("Spatial Test: ", spatial_test)
        print()


        # Calculate RMSSE for testing predictions
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_abs_diff = np.mean(np.abs(test_data[1:] - test_data[:-1]))
        test_rmsse = test_rmse / test_abs_diff

        print('Test RMSE:', test_rmse)
        print('Test RMSSE:', test_rmsse)
        print()
    
        print("Homogeneous(Should be True):")
        homogenous = self.is_homogeneous(data)
        print(homogenous)

        if show_graph:
            # NOTE: +1 Bc data is not stripped in PriceModel
            days_train = [total_data_dict["Dates"][int(i+split)] for i in range(y_test.shape[0])]
            # Plot the actual and predicted prices
            plt.figure(figsize=(18, 6))

            predicted_test = plt.plot(days_train, test_predictions, label='Predicted Test')
            actual_test = plt.plot(days_train, y_test, label='Actual Test')

            plt.title(f'{stock_symbol} {title}')
            plt.xlabel(x_label)
            plt.ylabel(y_label)

            import matplotlib.ticker as ticker
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(7))

            plt.legend(
                [predicted_test[0], actual_test[0]],#[real_data, actual_test[0], actual_train],
                ['Predicted Test', 'Actual Data']#['Real Data', 'Actual Test', 'Actual Train']
            )
            plt.show()
        return directional_test, spatial_test, test_rmse, test_rmsse, homogenous

    def load(self, name: Optional[str]=None):
        """
        This method will load the model using the tensorflow load method.

        Returns:
            None: If no model is loaded
            BaseModel: The saved model if it was successfully saved
        """
        if self.model:
            return None
        if not name:
            name = self.__class__.__name__

        self.model = load_model(f"Stocks/{self.stock_symbol}/{name}_model")
        try:
            with open(f"Stocks/{self.stock_symbol}/min_max_data.json", 'r') as file:
                self.scaler_data = json.load(file)
        except FileNotFoundError:
            pass

        # type: ignore[no-any-return]
        return self.model

    def indicators_past_num_days(self, stock_symbol: str, end_date: str,
                                 information_keys: List[str], scaler_data: Dict[str, int],
                                 cached_info: pd.DataFrame, num_days: int) -> Dict[str, Union[float, str]]:
        """
        This method will return the indicators for the past `num_days` days specified in the
        information keys. It will use the cached information to calculate the indicators
        until the `end_date`.

        Args:
            information_keys (List[str]): tells model the indicators to use
            scaler_data (Dict[str, int]): used to scale indicators
            cached_info (pd.DataFrame): The cached information
            num_days (int): The number of days to calculate the indicators for
        
        Returns:
            dict: A dictionary containing the indicators for the stock data
                Values will be floats except some expections tht need to be
                processed during run time
        """
        stock_data = {}

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
            earnings_dates, earnings_diff = get_earnings_history(stock_symbol)
            
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            date = end_datetime - relativedelta(days=num_days)

            stock_data['earnings dates'] = []
            stock_data['earning diffs'] = [] # type: ignore[attr]
            low = scaler_data['earning diffs']['min'] # type: ignore[index]
            diff = scaler_data['earning diffs']['diff'] # type: ignore[index]

            for i in range(num_days):
                if not end_date in earnings_dates:
                    stock_data['earning diffs'].append(0)
                    continue
                i = earnings_dates.index(date)
                scaled = (earnings_diff[i]-low) / diff
                stock_data['earning diffs'].append(scaled)
        if self.scaler_data is None:
            # Scale each column manually
            for column in information_keys:
                if column in non_daily:
                    continue
                low = scaler_data[column]['min'] # type: ignore[index]
                diff = scaler_data[column]['diff'] # type: ignore[index]
                column_values = stock_data[column]
                scaled_values = (column_values - low) / diff
                scaled_values = (column_values - low) / diff
                stock_data[column] = scaled_values
        return stock_data

    def update_cached_info_online(self):
        """
        updates `self.cached_info`

        information_keys is so you can update once to get all the info
        look at `loop_implementation` for reference
        """
        end_datetime = datetime.strptime(self.end_date, "%Y-%m-%d")

        #_________________ GET Data______________________#
        ticker = yf.Ticker(self.stock_symbol)
        cached_info = self.cached_info
        #NOTE: optimize bettween
        if cached_info is None:
            start_datetime = end_datetime - relativedelta(days=self.num_days*4+20)
            if 'ema_200' in self.information_keys:
                start_datetime = start_datetime - relativedelta(days=200)
            cached_info = ticker.history(start=start_datetime, interval="1d")
            if len(cached_info) == 0: # type: ignore[arg-type]
                raise ConnectionError("Stock data failed to load. Check your internet")
        else:
            start_datetime = end_datetime - relativedelta(days=1)
            day_info = ticker.history(start=start_datetime, end=self.end_date, interval="1d")
            if len(day_info) == 0: # type: ignore[arg-type]
                raise ConnectionError("Stock data failed to load. Check your internet")
            cached_info = cached_info.drop(cached_info.index[0])
            cached_info = pd.concat((cached_info, day_info))
        return cached_info

    def update_cached_online(self):
        """
        This method updates the cached data using the internet.
        """
        cached = self.indicators_past_num_days(
            self.stock_symbol, self.end_date,
            self.information_keys, self.scaler_data,
            self.cached_info, self.num_days
        )
        cached = [cached[key] for key in self.information_keys if is_floats(cached[key])]
        self.cached = np.transpose(cached)

    def update_cached_offline(self) -> None:
        """This method updates the cached data without using the internet."""
        warn("For Testing")

        end_date = self.end_date
        #_________________ GET Data______________________#
        if not self.cached_info:
            with open(f"Stocks/{self.stock_symbol}/info.json", 'r') as file:
                cached_info = json.load(file)

            if not self.end_date in cached_info['Dates']:
                raise ValueError("end is before or after `Dates` range")
            end_index = cached_info["Dates"].index(self.end_date)
            cached = []
            for key in self.information_keys:
                if key in non_daily_no_use:
                    continue
                cached.append(
                    cached_info[key][end_index-self.num_days:end_index]
                )
            self.cached = np.transpose(cached)
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
        
        Warning:
            It is better to do this in your own code so online and offline are split
        """
        warn('It is better to do this in your own code so online and offline are split')
        end_datetime = datetime.strptime(self.end_date, "%Y-%m-%d")

        start_datetime = end_datetime - relativedelta(days=1)
        nyse = get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start_datetime, end_date=end_datetime+relativedelta(days=2))
        if self.end_date not in schedule.index:
            return None

        try:
            if type(self.cached_info) is Dict:
                raise ConnectionError("It has already failed to lead")
            self.cached_info = self.update_cached_info_online(self.information_keys)
            self.update_cached_online()
        except ConnectionError as error1:
            warn("Stock data failed to download. Check your internet")
            if type(self.cached_info) is pd.DataFrame:
                self.cached_info = None
            try:
                self.update_cached_offline()
            except ValueError as error2:
                print('exception from online prediction: ', error1)
                print('exception from offline prediction: ', error2)
                raise RuntimeError('Neither the online or offline updating of `cached` worked')

        if self.cached is None:
            raise RuntimeError('Neither the online or offline updating of `cached` worked')

        date_object = datetime.strptime(self.start_date, "%Y-%m-%d")
        next_day = date_object + relativedelta(days=1)
        self.start_date = next_day.strftime("%Y-%m-%d")

        date_object = datetime.strptime(self.end_date, "%Y-%m-%d")
        next_day = date_object + relativedelta(days=1)
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

class PriceModel(BaseModel):
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

    def __init__(self, start_date: str = None,
                 end_date: Optional[Union[date, str]] = None,
                 stock_symbol: Optional[Union[date, str]] = "AAPL",
                 num_days: int = None,
                 information_keys: List[str]=["Close"]) -> None:
        super().__init__(start_date=start_date,
                       end_date=end_date,
                       stock_symbol=stock_symbol,
                       num_days=num_days,
                       information_keys=information_keys
                       )

    def process_x_y_total(self, x_total, y_total, num_days, time_shift):
        # NOTE: Strip last day for test
        x_total = x_total[:-1]
        y_total = y_total[:-1]
        return x_total, y_total

    def train(self, epochs: int = 1000, patience: int = 5, time_shift: int = 0, add_scaling: bool = True, add_noise: bool = True, use_transfer_learning: bool = False, test: bool = False, create_model: Callable[..., Any] = create_LSTM_model) -> None:
        return super().train(epochs, patience, time_shift, add_scaling, add_noise, use_transfer_learning, test, create_model)

    def profit(self, pred, prev):
        return pred/prev

class PercentageModel(BaseModel):
    """
    Different model that uses min-max scaling on data and accuracy as output. It handles the actual training, saving,
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

    def __init__(self, start_date: str = None,
                 end_date: Optional[Union[date, str]] = None,
                 stock_symbol: Optional[Union[date, str]] = "AAPL",
                 num_days: int = None,
                 information_keys: List[str]=["Close"]) -> None:
        if num_days is None:
            num_days = 10
        super().__init__(start_date=start_date,
                       end_date=end_date,
                       stock_symbol=stock_symbol,
                       num_days=num_days,
                       information_keys=information_keys
                       )
        self.cached_cached = None#(For stock caching on 4d data)

    def process_x_y_total(self, x_total, y_total, num_days, time_shift):
        # NOTE: Strips 1st day becuase -0 is 0. Look at `y_total[:-1]`
        y_total = y_total[1:] / y_total[:-1]
        y_total[np.isinf(y_total) | np.isnan(y_total)] = 1.0
        y_total -= 1.0
        y_total *= 100 #normal %

        if time_shift != 0:
            x_total = x_total[:-time_shift]
            y_total = y_total[time_shift:]
        num_windows = x_total.shape[0] - num_days + 1
        # Create a 3D numpy array to store the scaled data
        scaled_data = np.zeros((num_windows, num_days, x_total.shape[1], x_total.shape[2]))

        for i in range(x_total.shape[0]-num_days):
            # Get the data for the current window using the i-window_size approach
            window = x_total[i : i + num_days]
            #total 4218, 10 windows, num_days 10, indicators 7

            # Calculate the high and low close prices for the current window
            high_close = np.max(window, axis=0)
            low_close = np.min(window, axis=0)

            # Avoid division by zero if high_close and low_close are equal
            scale_denominator = np.where(high_close == low_close, 1, high_close - low_close)

            # Scale each column using broadcasting
            scaled_window = (window - low_close) / scale_denominator
            # Store the scaled window in the 3D array
            scaled_data[i] = scaled_window
        y_total = y_total[:-num_days+2]

        return scaled_data, y_total

    def train(self, epochs: int = 1000, patience: int = 5, time_shift: int = 0, add_noise: bool = True, use_transfer_learning: bool = False, test: bool = False, create_model: Callable[..., Any] = create_LSTM_model2) -> None:
        return super().train(epochs, patience, time_shift, False, add_noise, use_transfer_learning, test, create_model)

    def test(self, time_shift: int = 0, show_graph: bool = False) -> None:
        title: str = "Stock Change Prediction"
        x_label: str = ''
        y_label: str = 'Price Change in %'
        return super().test(time_shift, show_graph, title, x_label, y_label)

    def update_cached_offline(self) -> None:
        if self.cached_cached is not None:
            self.cached = self.cached_cached
        super().update_cached_offline()
        self.cached_cached = np.copy(self.cached)


        scaled_data = np.zeros((1, self.num_days, self.cached.shape[0], self.cached.shape[1]))

        # Get the data for the current window using the i-window_size approach
        window = self.cached#self.cached[-self.num_days:]

        # Calculate the high and low close prices for the current window
        high_close = np.max(window, axis=0)
        low_close = np.min(window, axis=0)
        # Avoid division by zero if high_close and low_close are equal
        scale_denominator = np.where(high_close == low_close, 1, high_close - low_close)

        # Scale each column using broadcasting
        scaled_window = (window - low_close) / scale_denominator
        # Store the scaled window in the 3D array
        scaled_data[0] = scaled_window
        self.cached = scaled_data

        #self.plot(self.cached[0][0])

    def profit(self, pred, prev):
        return pred

ImpulseMACD_indicators = ['Close', 'Histogram', 'Momentum', 'Change', 'ema_flips', 'signal_flips', '200-day EMA']
Reversal_indicators = ['Close', 'gradual-liquidity spike', '3-liquidity spike', 'momentum_oscillator']
Earnings_indicators = ['Close', 'earnings dates', 'earning diffs', 'Momentum']
RSI_indicators = ['Close', 'RSI', 'TRAMA']
break_out_indicators = ['Close', 'Bollinger Middle',
    'Above Bollinger', 'Bellow Bollinger', 'Momentum']
super_trends_indicators = ['Close', 'supertrend1', 'supertrend2',
    'supertrend3', '200-day EMA', 'kumo_cloud']


def update_transfer_learning(model: BaseModel,
                             companies: List= ["GE", "DIS", "AAPL", "GOOG", "META"]
                             ) -> None:
    """Updates Tranfer Learning Model"""
    model.end_date = date.today()-relativedelta(days=30)
    use = False
    for company in companies:
        model.stock_symbol = company
        model.update_dates()
        model.end_date = date.today()-relativedelta(days=30)
        model.train(use_transfer_learning=use)
        model.save(transfer_learning=True)

        if not use:
            use = True
    model.stock_symbol = "AMZN"
    model.update_dates()
    model.end_date = date.today()-relativedelta(days=30)
    model.train(test=True)
    model.test(show_graph=True)

if __name__ == "__main__":
    modelclass = PercentageModel
    indicators = [break_out_indicators]#[ImpulseMACD_indicators, Reversal_indicators, Earnings_indicators, break_out_indicators, super_trends_indicators]
    #indicators = list(set(chain(*indicators)))
    #indicators.remove('Close')
    #print(indicators)
    #indicators.remove('RSI')
    #indicators.insert(0, 'Close')
    #indicators = [indicators]
    test_models = []
    for company in ["AAPL", "HD", "DIS", "GOOG"]:
        model = modelclass(stock_symbol=company, information_keys=ImpulseMACD_indicators)
        #model.load()
        #model.stock_symbol = "T"
        model.start_date = "2006-03-24"
        model.end_date = "2023-03-24"
        model.num_days = 10

        model.train(epochs=1000, use_transfer_learning=False, test=True)
        model.save()
        #model.stock_symbol = "HD"
        model.start_date = "2020-04-11"
        model.end_date = "2023-04-11"
        test_models.append(model)

    for model in test_models:
        model.test(show_graph=True)
