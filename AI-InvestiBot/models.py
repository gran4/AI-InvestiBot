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
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
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
    all_indicators,
    suggested_strategies, suggested_companies, 
    check_for_holidays, get_relavant_values,
    create_sequences, non_daily_no_use, is_floats,
    calculate_percentage_movement_together,
    indicators_to_add_noise_to, indicators_to_scale,
    get_indicators_for_date
)


__all__ = (
    'CustomLoss',
    'CustomLoss2',
    'BaseModel',
    'PriceModel',
    'PercentageModel',
)

def update_dates(
        stock_symbol,
        start_date=None,
        end_date=None,
    ):
    with open(f'Stocks/{stock_symbol}/info.json', 'r') as file:
        Dates = json.load(file)['Dates']
    if start_date is None:
        start_date = Dates[0]
    if end_date is None:
        end_date = Dates[-1]

    if type(start_date) == date:
        start_date = start_date.strftime("%Y-%m-%d")
    if type(end_date) == date:
        end_date = end_date.strftime("%Y-%m-%d")
    return check_for_holidays(
        start_date, end_date
    )

def plot(total_data_dict: dict, split: int,
         test_predictions: np.ndarray, y_test: np.ndarray,
         stock_symbol: str, title: str,
         x_label: str, y_label: str) -> None:
    """Plots any np.array that you give in"""
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
        [predicted_test[0], actual_test[0]],
        ['Predicted Test', 'Actual Data']
    )
    plt.show()

def is_homogeneous(arr) -> bool:
        """Checks if any of the models indicators are missing"""
        return len(set(arr.dtype for arr in arr.flatten())) == 1

def check_early_stop(model, history, patience, patience_counter, best_val_loss, best_weights):
    current_val_loss = history.history['val_loss'][0]
    # Check for improvement
    early_stop_triggered = False
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        patience_counter = 0  # Reset patience since we've improved
        best_weights = model.get_weights() # Save the current best model weights to memory
    else:
        patience_counter += 1  # Increment patience counter
        # Early stopping check
        if patience_counter > patience:
            early_stop_triggered = True
    return early_stop_triggered, best_val_loss, best_weights

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
                 num_days: int = None, name: str = "model",
                 information_keys: List[str]=["Close"]) -> None:
        if num_days is None:
            with open(f'Stocks/{stock_symbol}/dynamic_tuning.json', 'r') as file:
                num_days = json.load(file)['num_days']

        self.information_keys = information_keys
        self.num_days = num_days
        self.name = name
        self.start_date, self.end_date = update_dates(stock_symbol, start_date=start_date, end_date=end_date)
        self.stock_symbol = stock_symbol

        self.model: Optional[Sequential] = None
        self.scale: bool = False
        self.scaler_data: Dict[str, float] = {}

#________For offline predicting____________#
        self.cached: Optional[np.ndarray] = None

        # NOTE: cached_info is a pd.DateFrame online,
        # while it is a Dict offline
        self.cached_info: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None

    def process_x_y_total(self, x_total, y_total, num_days, time_shift):
        return x_total, y_total

    def train(self, epochs: int=100,
              patience: int=5, time_shift: int=0,
              add_scaling: bool=False, add_noise: bool=True,
              use_transfer_learning: bool=False, test: bool=False,
              create_model: Callable=create_LSTM_model, loss: Loss=CustomLoss2()) -> None:
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

        #_________________ GET Data______________________#
        _, data, self.scaler_data = get_relavant_values(
            stock_symbol, information_keys, start_date=start_date, end_date=end_date
        )

        #_________________Process Data for LSTM______________________#
        split = int(len(data))
        if test:
            x_train, y_train = create_sequences(data[:int(split*.8)], num_days)
        else:
            x_train, y_train = create_sequences(data, num_days)
        x_train, y_train = self.process_x_y_total(x_train, y_train, num_days, time_shift)
        x_val, y_val = create_sequences(data[int(split*.8)-num_days:], num_days)
        x_val, y_val = self.process_x_y_total(x_val, y_val, num_days, time_shift)

        num_windows = x_train.shape[0]

        processed = [key for key in information_keys if not key in non_daily_no_use]
        if self.model:
            model = self.model
        elif len(x_train.shape) == 3:
            model = create_model((num_days, len(information_keys)), loss=loss)
        else:
            model = create_model((x_train.shape[1], num_days, len(information_keys)), loss=loss)

        if use_transfer_learning:
            transfer_model = load_model(f"transfer_learning_models/{self.name}_model")
            for layer_idx, layer in enumerate(model.layers):
                if layer.name in transfer_model.layers[layer_idx].name:
                    layer.set_weights(transfer_model.layers[layer_idx].get_weights())

        # Initialize monitoring variables
        best_val_loss = float('inf')  # Set initial best to infinity
        patience_counter = 0
        try:
            best_weights = model.get_weights()
        except:
            best_weights = None
        #_________________Train it______________________#
        epoch = 0
        while True:
            epoch += 1
            if epoch > epochs:
                break
            print(f'Starting Epoch {epoch}/{epochs}')

            # Standard training pass with EarlyStopping
            loss.focused_training = False
            history = model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val), verbose=1)
            early_stop_triggered, best_val_loss, best_weights = check_early_stop(model, history, patience, patience_counter, best_val_loss, best_weights)
            if early_stop_triggered:
                print("Early stopping triggered")
                # Restore the best weights from memory
                model.set_weights(best_weights)
                break
            elif epochs < 10:
                continue
            # Evaluate to identify mispredictions based on sign difference
            predictions = model.predict(x_train).flatten()
            sign_diff = np.sign(predictions) != np.sign(y_train)
            X_sign_diff = x_train[sign_diff]
            y_sign_diff = y_train[sign_diff]

            epoch += 1
            if epoch > epochs:
                break
            print(f'Starting Epoch {epoch}/{epochs}')

            loss.focused_training = True
            # Focused training on instances with sign differences, with EarlyStopping
            history = model.fit(X_sign_diff, y_sign_diff, epochs=1, validation_data=(x_val, y_val), verbose=1)
            early_stop_triggered, best_val_loss, best_weights = check_early_stop(model, history, patience, patience_counter, best_val_loss, best_weights)
            # Early stopping check
            if early_stop_triggered:
                print("Early stopping triggered")
                # Restore the best weights from memory
                model.set_weights(best_weights)
                break
        self.model = model

        return
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min', restore_best_weights=True)
        divider = int(split/2)
        if add_scaling:
            indices_cache = [information_keys.index(key) for key in indicators_to_scale if key in information_keys]

            x_train_copy = np.copy(x_train)
            y_train_copy = np.copy(y_train)
            model.fit(x_train_copy*1.1, y_train_copy*1.1, validation_data=(x_train_copy, y_train_copy), callbacks=[early_stopping], batch_size=64, epochs=epochs)

            x_train_p1 = np.copy(x_train[:divider])
            y_train_p1 = np.copy(y_train[:divider])
            x_train_p1[:, indices_cache] *= 2
            y_train_p1 *= 2
        if add_noise:
            x_train_copy = np.copy(x_train)
            y_train_copy = np.copy(y_train)
            # Get the indices of indicators to add noise to
            indices_cache = [information_keys.index(key) for key in indicators_to_add_noise_to if key in information_keys]

            # Create a noise array with the same shape as x_train's selected columns
            noise = np.random.uniform(-0.001, 0.001)
            # Add noise to the selected columns of x_train
            x_train_copy[:, indices_cache] += noise
            y_train_copy += np.random.uniform(-0.001, 0.001, size=y_train.shape[0])
            model.fit(x_train, y_train, validation_data=(x_train, y_train), callbacks=[early_stopping], batch_size=64, epochs=epochs)

        #Ties it together on the real data
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=[early_stopping], batch_size=64, epochs=epochs)
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
            self.model.save(f"transfer_learning_models/{name}_model")
            return
        self.model.save(f"Stocks/{self.stock_symbol}/{name}_model")

        if not self.scale: return
        if os.path.exists(f'Stocks/{self.stock_symbol}/min_max_data.json'):
            with open(f"Stocks/{self.stock_symbol}/min_max_data.json", 'r') as file:
                temp = json.load(file)
            self.scaler_data.update({key: value for key, value in temp.items()})

        with open(f"Stocks/{self.stock_symbol}/min_max_data.json", "w") as json_file:
            json.dump(self.scaler_data, json_file)

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
            stock_symbol, information_keys, scaler_data=self.scaler_data, start_date=start_date, end_date=end_date
        )
        #_________________Process Data for LSTM______________________#
        split = int(len(data) * 0.8)
        test_data = data[split-num_days-1:] # minus by `num_days` to get full range of values during the test period 
        x_test, y_test = create_sequences(test_data, num_days)
        x_test, y_test = self.process_x_y_total(x_test, y_test, num_days, time_shift)
        #_________________TEST QUALITY______________________#
        test_predictions = self.model.predict(x_test)

        # NOTE: This cuts data at the start to account for `num_days`
        if time_shift > 0:
            test_data = data[:-time_shift]
        directional_test, spatial_test, together_test = calculate_percentage_movement_together(y_test, test_predictions)
        print("Directional Test: ", directional_test)
        print("Spatial Test: ", spatial_test)
        print("Together Test(MOST important): ", together_test)
        print()

        print(len(y_test))
        print(len(test_predictions))
        #Calculate RMSSE for testing predictions
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_abs_diff = np.mean(np.abs(test_data[1:] - test_data[:-1]))
        test_rmsse = test_rmse / test_abs_diff

        print('Test RMSE:', test_rmse)
        print('Test RMSSE:', test_rmsse)
        print()
    
        print("Homogeneous(Should be True):")
        homogenous = is_homogeneous(data)
        print(homogenous)

        if show_graph:
            plot(total_data_dict, split, test_predictions, y_test, stock_symbol, title, x_label, y_label)
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

    def update_cached_info_online(self):
        """
        updates `self.cached_info`

        information_keys is so you can update once to get all the info
        look at `loop_implementation` for reference
        """
        # NOTE: ticker.history does not include the end date, so we correct it by adding one date to the end date
        end_datetime = datetime.strptime(self.end_date, "%Y-%m-%d")
        corrected_end_datetime = end_datetime + timedelta(days=1)
        corrected_end_datetime_str = corrected_end_datetime.strftime('%Y-%m-%d')

        #_________________ GET Data______________________#
        ticker = yf.Ticker(self.stock_symbol)
        cached_info = self.cached_info
        #NOTE: optimize bettween
        if cached_info is None:
            cached_info = ticker.history(start=self.start_date, end=corrected_end_datetime_str, interval="1d")
            if len(cached_info) == 0: # type: ignore[arg-type]
                raise ConnectionError("Stock data failed to load. Check your internet")
        else:
            start_datetime = end_datetime - relativedelta(days=1)
            day_info = ticker.history(start=start_datetime, end=corrected_end_datetime_str, interval="1d")
            if len(day_info) == 0: # type: ignore[arg-type]
                raise ConnectionError("Stock data failed to load. Check your internet")
            cached_info = cached_info.drop(cached_info.index[0])
            cached_info = pd.concat((cached_info, day_info))
        return cached_info

    def update_cached_online(self):
        """
        This method updates the cached data using the internet.
        """
        cached = get_indicators_for_date(
            self.stock_symbol, self.end_date,
            self.information_keys,
            self.cached_info, self.num_days, scale=self.scale, scaler_data=self.scaler_data
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

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "stock_symbol":
            try:
                self.start_date, self.end_date = update_dates(value, start_date=self.start_date, end_date=self.end_date)
            except AttributeError as e:
                print(e)

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
                 num_days: int = None, name: str = "model",
                 information_keys: List[str]=["Close"]) -> None:
        super().__init__(start_date=start_date,
                       end_date=end_date,
                       stock_symbol=stock_symbol,
                       num_days=num_days, name=name,
                       information_keys=information_keys
                       )
        self.scale: bool = True

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
                 num_days: int = None, name: str = "model",
                 information_keys: List[str]=["Close"]) -> None:
        if num_days is None:
            num_days = 10
        super().__init__(start_date=start_date,
                       end_date=end_date,
                       stock_symbol=stock_symbol,
                       num_days=num_days, name=name,
                       information_keys=information_keys
                       )
        self.cached_cached = None #(For stock caching on 4d data)

    def process_x_y_total(self, x_total, y_total, num_days, time_shift):
        # NOTE: Strips 1st day becuase -0 is 0. Look at `y_total[:-1]`
        if time_shift != 0:
            x_total = x_total[:-time_shift]
            y_total = (y_total[time_shift+1:] / y_total[:-time_shift-1] - 1.0) * 100
        else:
            y_total = (y_total[1:] / y_total[:-1] - 1.0) * 100
        x_total = x_total[1:]
        return x_total, y_total
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

        y_total = y_total[:-num_days+1]

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
        window = self.cached

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

    def profit(self, pred, prev):
        return pred

def update_transfer_learning(model: BaseModel,
                             companies: List= ["KO", "AAPL", "GOOG", "NVDA", "NKE", "AMZN", "MSFT"],
                             ) -> None:
    """Updates Tranfer Learning Model"""
    model.end_date = date.today()-relativedelta(days=30)
    for company in companies:
        model.stock_symbol = company
        model.start_date, model.end_date = update_dates(company)
        model.end_date = date.today()-relativedelta(days=30)
        model.train(patience=0)

    model.save(name=model.name, transfer_learning=True)


if __name__ == "__main__": # PROBLEM in the model itself during training. Probably the dataer4
    modelclass = PercentageModel

    # for name, indicators in suggested_strategies.items():
    #    model = PercentageModel(information_keys=indicators, name=name)
    #    model.num_days = 7
    #    update_transfer_learning(model)

    model = modelclass(information_keys=all_indicators, name='all_in')
    for company in suggested_companies:
        model.stock_symbol = company
        model.start_date, model.end_date = update_dates(company)
        model.num_days = 7
        print(model.stock_symbol, model.start_date, model.end_date)
        model.train(epochs=10, patience=1, time_shift=0, use_transfer_learning=False, test=True)
    model.stock_symbol = "BRK-B"
    model.start_date, model.end_date = update_dates(company)
    model.test(show_graph=True, time_shift=0)


"""     test_models = []
    for company in suggested_companies:
        for name, indicators in suggested_strategies.items():
            model = modelclass(stock_symbol=company, information_keys=indicators, name=name)
            model.num_days = 7
            model.train(epochs=10, patience=0, time_shift=0, use_transfer_learning=False, test=True)
            test_models.append(model)
    for model in test_models:
        model.test(show_graph=True, time_shift=0)

 """