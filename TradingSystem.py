from abc import ABC, abstractmethod
from threading import Thread
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from Tradingfuncs import create_sequences


class TradingSystem(ABC):
    def __init__(self, api, symbol, time_frame, system_id, system_label):
        self.api = api
        self.symbol = symbol
        self.time_frame = time_frame
        self.system_id = system_id
        self.system_label = system_label
        thread = Thread(target=self.system_loop)
        thread.start()

    @abstractmethod
    def buy(self):
        pass

    @abstractmethod
    def sell(self):
        pass

    @abstractmethod
    def loop(self):
        pass

    @abstractmethod
    def create_sequences(data, num_days):
        pass
        
class DayTrader():
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-09",
                 stock_symbol: str = "APPL") -> None:
        # Use yfinance to fetch the stock data from Yahoo Finance
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

        num_days = 60

        # Preprocess the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

        # Split the data into training and testing sets
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]


        X_total, Y_total = create_sequences(train_data, num_days)
        X_train, Y_train = create_sequences(train_data, num_days)
        X_test, Y_test = create_sequences(test_data, num_days)

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(num_days, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_total, Y_total, batch_size=32, epochs=20)
        model.fit(X_test, Y_test, batch_size=32, epochs=20)
        model.fit(X_train, Y_train, batch_size=32, epochs=20)


