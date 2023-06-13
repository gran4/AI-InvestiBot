from typing import Optional

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

    def buy(self, amount, ticker):
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
