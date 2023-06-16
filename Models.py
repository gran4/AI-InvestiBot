import json

import numpy as np
import pandas as pd

from typing import Optional, List, Dict
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import LSTM, Dense

from Tradingfuncs import *
from getInfo import calculate_momentum_oscillator, get_liquidity_spikes, get_earnings_history
from warnings import warn

class BaseModel(object):
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-05",
                 stock_symbol: str = "AAPL", information_keys: List=[]) -> None:
        print("EDJNDENJDENJDE")
        self.start_date = start_date
        self.end_date = end_date
        self.stock_symbol = stock_symbol
        self.information_keys = information_keys

        self.cached: Optional[pd.DataFrame] = None
        self.model: Optional[Sequential] = None
        self.data: Optional[Dict] = None

    def train(self):
        warn("If you saved before, use load func instead")

        start_date = self.start_date
        end_date = self.end_date
        stock_symbol = self.stock_symbol
        information_keys = self.information_keys

        #_________________ GET Data______________________#
        num_days = 60
        data, start_date, end_date = get_relavant_Values(start_date, end_date, stock_symbol, information_keys)
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

        self.model = model

    def save(self):
        #_________________Save Model______________________#
        # Save structure to json
        jsonversion = self.model.to_json()
        with open(f"{self.stock_symbol}/model.json", "w") as json_file:
            json_file.write(jsonversion)

        # Save weights to HDF5
        self.model.save_weights(f"{self.stock_symbol}/weights.h5")

        with open(f"{self.stock_symbol}/data.json", "w") as json_file:
            json.dump(self.data, json_file)

    def test(self):
        """EXPENSIVE"""
        warn("Expensive, for testing purposes")

        if not self.model:
            raise LookupError("Compile or load model first")

        stock_symbol = self.stock_symbol
        information_keys = self.information_keys

        #_________________ GET Data______________________#
        num_days = 60
        data, start_date, end_date = get_relavant_Values(start_date, end_date, stock_symbol, information_keys)


        #_________________Process Data for LSTM______________________#
        # Split the data into training and testing sets
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

        X_train, Y_train = create_sequences(train_data, num_days)
        X_test, Y_test = create_sequences(test_data, num_days)

        #_________________TEST QUALITY______________________#
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)

        # Calculate RMSSE for training and testing predictions
        train_rmsse = np.sqrt(mean_squared_error(Y_train, train_predictions)) / np.mean(Y_train[1:] - Y_train[:-1])
        test_rmsse = np.sqrt(mean_squared_error(Y_test, test_predictions)) / np.mean(Y_test[1:] - Y_test[:-1])

        print('Train RMSSE:', train_rmsse)
        print('Test RMSSE:', test_rmsse)

    def load(self):
        if self.model:
            return
        with open(f"{self.stock_symbol}/model.json") as file:
            model_json = file.read()
        model = model_from_json(model_json)
        model.load_weights(f'{self.stock_symbol}/weights.h5')

        with open(f"{self.stock_symbol}/data.json") as file:
            self.data = file.read()

        self.model = model
        return model

    def getInfoToday(self, period: int=14) -> List[float]:
        """
        Similar to getHistoricalData but it only gets today
        
        cached_data used so less data has to be gotten from yf.finance
        """
        #Limit attribute look ups + Improve readability
        stock_symbol = self.stock_symbol
        information_keys = self.information_keys
        end_date = self.end_date
        cached_data = self.cached

        #_________________ GET Data______________________#
        ticker = yf.Ticker(stock_symbol)
        if cached_data:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            day_data = ticker.history(start=end_date, end=end_date, interval="1d")
            cached_data = cached_data.drop(cached_data.index[0])
            cached_data.append(day_data, ignore_index=True)
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = end_date - timedelta(days=200)
            cached_data = ticker.history(start=start_date, end=end_date, interval="1d")

        #_________________MACD Data______________________#
        # Calculate start and end dates
        cached_data['12-day EMA'] = cached_data['Close'].ewm(span=12, adjust=False).mean().iloc[-1]

        # 26-day EMA
        cached_data['26-day EMA'] = cached_data['Close'].ewm(span=26, adjust=False).mean().iloc[-1]

        # MACD
        cached_data['MACD'] = cached_data['12-day EMA'] - cached_data['26-day EMA']

        # Signal Line
        span = 9
        signal_line = cached_data['MACD'].rolling(window=span).mean().iloc[-1]
        cached_data['Signal Line'] = signal_line

        # Histogram
        cached_data['Histogram'] = cached_data['MACD'] - cached_data['Signal Line']

        # 200-day EMA
        cached_data['200-day EMA'] = cached_data['Close'].ewm(span=200, adjust=False).mean().iloc[-1]

        # Basically Impulse MACD
        cached_data['Change'] = cached_data['Close'].diff().iloc[-1]
        cached_data['Momentum'] = cached_data['Change'].rolling(window=10, min_periods=1).sum().iloc[-1]

        # Breakout Model
        cached_data["Gain"] = cached_data["Change"] if cached_data["Change"] > 0 else 0
        cached_data["Loss"] = abs(cached_data["Change"]) if cached_data["Change"] < 0 else 0
        cached_data["Avg Gain"] = cached_data["Gain"].rolling(window=14, min_periods=1).mean().iloc[-1]
        cached_data["Avg Loss"] = cached_data["Loss"].rolling(window=14, min_periods=1).mean().iloc[-1]
        cached_data["RS"] = cached_data["Avg Gain"] / cached_data["Avg Loss"]
        cached_data["RSI"] = 100 - (100 / (1 + cached_data["RS"]))

        # TRAMA
        volatility = cached_data['Close'].diff().abs().iloc[-1]
        trama = cached_data['Close'].rolling(window=period).mean().iloc[-1]
        cached_data['TRAMA'] = trama + (volatility * 0.1)

        # Reversal
        cached_data['gradual-liquidity spike'] = get_liquidity_spikes(cached_data['Volume'], gradual=True).iloc[-1]
        cached_data['3-liquidity spike'] = get_liquidity_spikes(cached_data['Volume'], z_score_threshold=4).iloc[-1]
        cached_data['momentum_oscillator'] = calculate_momentum_oscillator(cached_data['Close']).iloc[-1]


        #_________________12 and 26 day Ema flips______________________#
        temp12 = cached_data['Close'].ewm(span=12, adjust=False).mean().iloc[-2]
        temp26 = cached_data['Close'].ewm(span=26, adjust=False).mean().iloc[-2]
        temp_bool = temp12 > temp26
        cached_data['flips'] = temp_bool != cached_data['12-day EMA'] > cached_data['26-day EMA']
        cached_data['flips'] = int(cached_data['flips'])

        #earnings stuffs
        earnings_dates, earnings_diff = get_earnings_history(stock_symbol)
        if not end_date in earnings_dates:
            cached_data['earnings diff'] = 0
        else:
            cached_data['earnings diff'] = earnings_diff
        cached_data['earnings dates'] = end_date

        # Scale them 0-1
        excluded_values = ['Date']  # Add any additional columns you want to exclude from scaling
        for info in cached_data.keys():
            if info in excluded_values:
                continue
            cached_data[info] = scale(cached_data[info], self.data)
        #NOTE: 'Dates' is irrelevant
        return [cached_data[key] for key in information_keys]

    def predict(self, info: Optional[List[float]] = None):
        """Wraps the models predict func"""
        if not info:
            info = self.getInfoToday()
        if self.model:
            return self.model.predict(info)
        raise LookupError("Compile or load model first")



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


class BreakoutModel(BaseModel):
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
    model = DayTradeModel()
    model.train()
    model.save()


    model = MACDModel()
    model.train()
    model.save()
    model = ImpulseMACDModel()
    model.train()
    model.save()
    model = ReversalModel()
    model.train()
    model = model.save()
    EarningsModel()
    model.train()
    model = model.save()
    BreakoutModel()
    model.train()
    model = model.save()

