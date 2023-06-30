"""
Name:
    trading_system.py

Purpose:
    This module provides a trading system that can be used to trade stocks.

Author:
    Grant Yul Hur

See also:
    Other modules related to running the stock bot -> lambda_implementation, loop_implementation
"""

from math import floor
from models import MACDModel
from trading_funcs import company_symbols


class DayTrader():
    """
    This is the base class for the DayTrader. Currently it only initalises
    a hard coded start date, end date, and stock symbol. It will be used to
    trade stocks.

    Args:
        start_date (str): The start date of the trading system
        end_date (str): The end date of the trading system
        stock_symbol (str): The stock symbol of the stock you want to trade
    """
    def __init__(self, start_date: str = "2020-01-01",
                 end_date: str = "2023-06-09",
                 stock_symbol: str = "AAPL") -> None:
        pass

# NOTE: Change to use amount of stock not money in stock.

all_models = {}#???????????? USE?????
holdings = {}
def update_all(model, manager):
    """
    This function will be used to update all the models and holdings.
    It can be used to create predictions and sort them. From these
    predictions it could determine whether it should buy or sell the stocks.

    Args:
        model (BaseModel): The model that will be used to predict the stock prices
        manager (ResourceManager): The manager that will be used to buy and sell stocks
    """
    #predicts tommorrows prices and sorts it
    predictions = {}
    for company_ticker in company_symbols:
        prediction = model.predict()####
        predictions[company_ticker] = prediction
    predictions = predictions.sort() # NOTE: FIX

    #Sell unprofitable ones
    vals = holdings.values()
    length = len(vals)

    keys = holdings.values()
    for holding, amount in zip(vals, keys):
        #if the index is too low, or the prediction is negative
        if keys.index(holding) < 5+length or prediction[company_ticker] < 0:
            manager.sell(amount, holding)

    #Buy new more profitable ones
    for company_ticker in predictions.keys():
        use = manager.check()
        #If there is not enough money to use
        if use <= 5 or prediction[company_ticker] < 0:
            break
        if stock_price > use:
            continue
        stocks = floor(use/stock_price)
        holdings[company_ticker] = stocks
        manager.buy(stocks, company_ticker)
