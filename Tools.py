
class ResourceManager(object):
    """
    Manages you money
    
    Put restraints on you money.
    Or type(bitcoin v stock)
    Or ...
    """
    def __init__(self, money, max_percent_used=100, max_used=None, max_in_one_stock=100):
        self.total_money = money
        if not max_percent_used:
            self.max_percent_used = 100
        if not max_used:
            self.max_percent_used = money
        self.max_in_one_stock = max_in_one_stock

        self.stock_mapping = {}

    def check(money: float, stock: str):
        """
        Returns how much can be used

        Returns how much can be used 
        with this Stock without going over 
        max_used, max_percent, or max_in_one_stock
        """
        total = self.money_used+money
        if total > self.max_used:
            return total-max_used
        elif total/self.money > self.max_percent_used:
            percent_acceptable = total/self.money
            percent_acceptable -= self.max_percent_used

            amount_acceptable = percent_acceptable*self.total_money
            return amount_acceptable
        elif stock in self.stock_mapping and self.stock_mapping[stock]+money > self.max_in_one_stock:
            return self.stock_mapping[stock]+money-self.max_in_one_stock
        #Has to be ok to use
        return money


class DayTrader(object):
    def __init__(self):
        pass