import json

import numpy as np
import pandas as pd

from typing import Optional, List, Dict
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

from Tradingfuncs import *
from getInfo import calculate_momentum_oscillator, get_liquidity_spikes, get_earnings_history
from warnings import warn
from copy import deepcopy

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
        model.fit(X_total, Y_total, batch_size=32, epochs=10)
        model.fit(X_test, Y_test, batch_size=32, epochs=10)
        model.fit(X_train, Y_train, batch_size=32, epochs=10)

        self.model = model

    def save(self):
        #_________________Save Model______________________#
        self.model.save(f"{self.stock_symbol}/model")

        with open(f"{self.stock_symbol}/data.json", "w") as json_file:
            json.dump(self.data, json_file)

    def test(self):
        """EXPENSIVE"""
        warn("Expensive, for testing purposes")

        if not self.model:
            raise LookupError("Compile or load model first")
        
        start_date = self.start_date
        end_date = self.end_date
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
        print(X_train.shape)
        print(X_test.shape)

        #_________________TEST QUALITY______________________#
        train_predictions = self.model.predict(train_data)
        test_predictions = self.model.predict(test_data)
        print(train_predictions)
        print(train_data.shape)


        # Calculate RMSSE for training and testing predictions
        train_rmse = np.sqrt(mean_squared_error(train_data, train_predictions))
        train_abs_diff = np.mean(np.abs(train_data[1:] - train_data[:-1]))
        train_rmsse = train_rmse / train_abs_diff

        test_rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
        test_abs_diff = np.mean(np.abs(test_data[1:] - test_data[:-1]))
        test_rmsse = test_rmse / test_abs_diff

        print('Train RMSSE:', train_rmsse)
        print('Test RMSSE:', test_rmsse)

    def load(self):
        if self.model:
            return
        self.model = load_model(f"{self.stock_symbol}/model")

        with open(f"{self.stock_symbol}/data.json") as file:
            self.data = json.load(file)

        return self.model

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
        stock_data = {}

        #_________________ GET Data______________________#
        ticker = yf.Ticker(stock_symbol)
        if cached_data:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            day_data = ticker.history(start=end_date, end=end_date, interval="1d")
            cached_data = cached_data.drop(cached_data.index[0])
            cached_data.append(day_data, ignore_index=True)
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = end_date - timedelta(days=260)
            cached_data = ticker.history(start=start_date, end=end_date, interval="1d")

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

        # Histogram
        stock_data['Histogram'] = stock_data['MACD'] - stock_data['Signal Line']

        # 200-day EMA
        stock_data['200-day EMA'] = cached_data['Close'].ewm(span=200, adjust=False).mean().iloc[-60:]

        # Basically Impulse MACD
        stock_data['Change'] = cached_data['Close'].diff().iloc[-60:]
        stock_data['Momentum'] = stock_data['Change'].rolling(window=10, min_periods=1).sum().iloc[-60:]

        # Breakout Model
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
        #last day's EMA
        temp = []
        ema12=cached_data['Close'].ewm(span=12, adjust=False).mean()
        ema26=cached_data['Close'].ewm(span=26, adjust=False).mean()
        shortmore = None
        for i in range(60, 0):
            short = ema12.iloc[-i]
            mid = ema26.iloc[-i]
            if shortmore is None:
                shortmore = short>mid
            elif shortmore and short<mid:
                temp.append(True)
                shortmore = False
                continue
            elif not shortmore and short>mid:
                temp.append(True)
                shortmore = True
                continue
            temp.append(False)
        stock_data['flips'] = list(map(int, temp))

        #earnings stuffs
        earnings_dates, earnings_diff = get_earnings_history(stock_symbol)
        all_dates = []
        #date = datetime.strptime(end_date, "%Y-%m-%d")
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

    def predict(self, info: Optional[List[float]] = None):
        """Wraps the models predict func"""
        if not info:
            info = self.getInfoToday()
        if self.model:
            #X_train, Y_train = create_sequences(info, 60)
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
    model.load()
    model.test()
    import time
    time.sleep(132)


    model = MACDModel()
    model.train()
    model.save()
    model = ImpulseMACDModel()
    model.train()
    model.save()
    model = ReversalModel()
    model.train()
    model = model.save()
    model = EarningsModel()
    model.train()
    model = model.save()
    model = BreakoutModel()
    model.train()
    model = model.save()

