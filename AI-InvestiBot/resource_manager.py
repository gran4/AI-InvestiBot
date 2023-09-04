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

from alpaca_trade_api import REST

__all__ = (
    'ResourceManager',
)


class ResourceManager:
    """
    This is used to manage the money you have, buy/sell stock,
    and place limits on how much you spend/ your risk.

    Args:
        Money (int): Money that you have
        max_percent (float): Max percent of money you can use for a single stock
        max (float): Max amount of money you can use for a stock
        stock_to_money_ratio (float): 0-1, 1 is all stock, 0 is all cash
        api_key (str): api key to use
        secret_key (str): secret key to use
        base_url (str): url decides whether of not it is
            paper(pretend) trading


    Put restraints on your money.
    """
    def __init__(self,
                 maximum: float=float("inf"),
                 max_percent: float=100.0,
                 stock_to_money_ratio: float= 1.0,
                 api_key: str = "",
                 secret_key: str = "",
                 base_url: str = "https://paper-api.alpaca.markets"
                 ) -> None:
        self.used = 0
        self.max_percent = max_percent
        self.max_per = maximum
        self.ratio = stock_to_money_ratio

        self.stock_mapping = {}
        self.api = REST(api_key, secret_key, base_url=base_url)

        account = self.api.get_account()
        cash = float(account.equity) - float(account.buying_power)
        buying_power = float(account.buying_power)

        # Calculate the total value of your account (including stock)
        self.total = cash + buying_power

    def check(self, symbol: str, balance: Optional[float]=None) -> float:
        """
        The purpose of this method is to determine how much money can be used
        for a stock. It takes into account the max, max_percent, and ratio to
        ensure that it does not exceed these metrics.

        Args:
            stock (str): The stock ticker
            balance (float): The amount of money you want to use
        
        Returns:
            int: The amount of stock you can buy
        """
        # Get account information
        account = self.api.get_account()
        if balance is None:
            open_orders = self.api.list_orders(status='open')

            # Calculate the total cost of pending orders
            total_pending_cost = 0

            for order in open_orders:
                last_trade = self.api.get_latest_trade(order.symbol)
                qty = float(order.qty)-float(order.filled_qty)
                print(order.qty, order.filled_qty, last_trade.price)
                total_pending_cost += qty * last_trade.price
            # Calculate available cash (subtracting funds reserved for pending orders)
            balance = float(account.cash) - total_pending_cost
            print(balance)
            print(balance)
            print(balance)
            print(balance)
            print(balance)
            print(balance)
            print(balance)
        # Get the current market price
        market_price = float(self.api.get_latest_trade(symbol).price)

        max_percent_in_stock = self.max_percent/100*balance/market_price

        # Calculate the maximum quantity to buy based on the stock-to-money ratio
        max_qty_based_on_ratio = int(balance / market_price * self.ratio)

        # Apply additional constraints
        max_qty = min(max_qty_based_on_ratio, self.max_per, max_percent_in_stock)

        return max_qty

    def buy(self, ticker: str, amount: Optional[int] = None) -> None:
        """
        This method will allow you to purchase a stock.

        Args:
            ticker (str): The stock ticker
            amount (Optional[int]): The amount of stock you want to buy,
                if None, the `Resource` can calculate it for you
        """
        if amount is None:
            amount = int(self.check(ticker))
        if amount <= 0:
            return
        self.api.submit_order(
            symbol=ticker,
            qty=amount,
            side='buy',
            type='market',
            time_in_force='day',
        )

    def sell(self, amount: Optional[int], ticker: str) -> None:
        """
        This method will allow you to sell a stock.

        Args:
            amount (int): The amount of stock you want to sell
            ticker (str): The stock ticker
        """
        self.api.submit_order(
            symbol=ticker,
            qty=amount,
            side='sell',
            type='market',
            time_in_force='day',
        )

    def is_in_portfolio(self, symbol):
        # Get positions
        positions = self.api.list_positions()

        # Check if the stock is in the portfolio
        for position in positions:
            if position.symbol == symbol:
                return True

        return False

