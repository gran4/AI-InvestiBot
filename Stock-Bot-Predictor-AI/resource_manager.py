"""
Name:
    resource_manager.py

Purpose:
    This module provides utility functions for stock trading. At the moment
    it only contains a resource_manager class. This class is used to manage
    your money and controlling how much you can use for a stock. 

Author:
    Grant Yul Hur

See also:
    Other modules related to running the stock bot -> lambda_implementation, loop_implementation
"""

from typing import Optional

__all__ = (
    'ResourceManager',
)


class ResourceManager:
    """
    This is used to manage the money you have, buy/sell stock,
    and place limits on how much you spend/ your risk.

    Args:
        Money (int): Money that you have
        max_percent (float): Max percent of money you can use for a stock
        max (float): Max amount of money you can use for a stock
        stock_to_money_ratio (float): 0-1, 1 is all stock, 0 is all cash

    Put restraints on your money.
    """
    def __init__(self, money: float, maximum: Optional[float]=None,
                 max_percent: float=100.0, max: float=1000.0,
                 stock_to_money_ratio: float= 1.0) -> None:
        self.total = money
        self.used = 0
        if not max_percent:
            self.max_percent = 100
        if not maximum:
            maximum = money
        self.max_used = maximum
        self.ratio = stock_to_money_ratio

        self.stock_mapping = {}
        self.api = None

    def check(self, stock: str, money: Optional[float]=None) -> float:
        """
        The purpose of this method is to determine how much money can be used
        for a stock. It takes into account the max, max_percent, and ratio to
        ensure that it does not exceed these metrics.

        Args:
            stock (str): The stock ticker
            money (float): The amount of money you want to use
        
        Returns:
            float: The amount of money you can use
        """
        if not money:
            money = self.total - self.used
        amount_acceptable = money

        total = self.used+money
        if total/self.total > self.max_percent:
            percent_acceptable = total/money
            percent_acceptable -= self.max_percent
            percent_acceptable *= self.total
            amount_acceptable = min(amount_acceptable, percent_acceptable)

        if stock in self.stock_mapping and self.stock_mapping[stock]+money > self.max_used:
            temp = self.stock_mapping[stock]+money-self.max_used
            #get lowest amount acceptable
            amount_acceptable = min(amount_acceptable, temp)

        temp = amount_acceptable+self.used
        if temp/self.total > self.ratio:
            amount_acceptable = self.total * self.ratio

        #Has to be ok to use
        return amount_acceptable

    def buy(self, amount: int, money: float, ticker: str) -> None:
        """
        This method will allow you to purchase a stock.

        Args:
            amount (int): The amount of stock you want to buy
            money (float): The amount of money you want to use
            ticker (str): The stock ticker
        """
        #it doesn't update so it is reset every time it is sold
        if ticker in self.stock_mapping:
            self.stock_mapping[ticker] = amount
        else:
            self.stock_mapping[ticker] += amount
        self.used += money

        self.api.submit_order(
                    symbol=ticker,
                    qty=amount,
                    side='buy',
                    type='market',
                    time_in_force='day',
                )

    def sell(self, amount: int, money: float, ticker: str) -> None:
        """
        This method will allow you to sell a stock.

        Args:
            amount (int): The amount of stock you want to sell
            money (float): The amount of money you want to use
            ticker (str): The stock ticker
        """
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
