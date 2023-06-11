from typing import Optional
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from Tradingfuncs import create_sequences



class ResourceManager(object):
    """
    Manages you money
    
    Args:
        Money int: money that you have
        max_percent float: max percent of money you can use for a stock
        max float: max amount of money you can use for a stock
        stock_to_money_ratio float: 0-1, 1 is all stock, 0 is all cash


    Put restraints on your money.
    """
    def __init__(self, money, max_percent=100, max=1000, stock_to_money_ratio = 1):
        self.total = money
        self.used = 0
        if not max_percent:
            self.max_percent = 100
        if not max:
            self.max_percent = money
        self.max = max
        self.ratio = stock_to_money_ratio

        self.stock_mapping = {}
        self.api = None

    def check(self, stock: str, money: Optional[float]=None):
        """
        Returns how much can be used

        Returns how much can be used 
        with this Stock without going over 
        max, max_percent, or ratio
        """
        if not money:
            money = self.total - self.used
        amount_acceptable = money

        total = self.used+money
        if total/self.total > self.max_percent:
            percent_acceptable = total/self.money
            percent_acceptable -= self.max_percent

            #Get lowest amount acceptable
            if amount_acceptable > percent_acceptable*self.total:
                amount_acceptable = percent_acceptable*self.total
            
        if stock in self.stock_mapping and self.stock_mapping[stock]+money > self.max:
            temp = self.stock_mapping[stock]+money-self.max
            #get lowest amount acceptable
            if amount_acceptable > temp:
                amount_acceptable = temp

        temp = amount_acceptable+self.used
        if temp/self.total > self.ratio:
            amount_acceptable = self.total * self.ratio

        #Has to be ok to use
        return amount_acceptable

    def buy(self, amount, money, ticker):
        #it doesn't update so it is reset every time it is sold
        if ticker in self.stock_mapping:
            self.stock_mapping[ticker] = money
        else:
            self.stock_mapping[ticker] += money
        self.used += money

        self.api.submit_order(
                    symbol=ticker,
                    qty=amount,
                    side='buy',
                    type='market',
                    time_in_force='day',
                )

    def sell(self, amount, money, ticker):
        #0 bc I want to reset it. Since, it doesn't update
        self.stock_mapping[ticker] = 0
        self.used -= money

        self.api.submit_order(
                    symbol=ticker,
                    qty=amount,
                    side='sell',
                    type='market',
                    time_in_force='day',
                )

class DayTradeModel(object):
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

        # Save structure to json
        jsonversion = model.to_json()
        with open(f"{stock_symbol}/model.json", "w") as json_file:
            json_file.write(jsonversion)

        # Save weights to HDF5
        model.save_weights(f"{stock_symbol}/weights.h5")


