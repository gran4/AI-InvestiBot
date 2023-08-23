from models import *

from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import json

def test_many(model_class: BaseModel=PercentageModel, strategy=['Close'], tests: int=20, *args, **kwargs):
    averages = []
    for i in range(tests):
        model = model_class(strategy=strategy)
        model.train(*args, **kwargs)
        if 'time_shift' in kwargs:
            results = model.test(time_shift = kwargs['time_shift'])
        else:
            results = model.test()
        averages.append(results)
    num_columns = len(averages[0])
    column_sums = [0] * num_columns
    for row in averages:
        for i, value in enumerate(row):
            column_sums[i] += value

    return [sum / len(averages) for sum in column_sums]


def test_individual_indepth(model: BaseModel=PercentageModel()):
    model.stock_symbol = "SBUX"
    with open(f"Stocks/{model.stock_symbol}/info.json", 'r') as file:
        cached_info = json.load(file)
    with open(f'Stocks/{model.stock_symbol}/dynamic_tuning.json', 'r') as file:
        relevant_years = json.load(file)['relevant_years']
    print(datetime.strptime(cached_info['Dates'][-1], "%Y-%m-%d")-relativedelta(years=relevant_years))
    model.start_date = datetime.strptime(cached_info['Dates'][-1], "%Y-%m-%d")-relativedelta(years=relevant_years)
    model.end_date = model.start_date+relativedelta(days=200+model.num_days)
    model.update_dates(model.start_date, model.end_date)
    print(type(model.start_date))
    #model.load()

    money_made = 0
    stocks_hold = 0
    bought_at = []

    signals = {}
    time_stamp = 0
    while True:
        if model.end_date == "2023-06-05":
            break
        print(money_made, model.end_date)
        import time
        time.sleep(.2)

        model.update_cached_offline()
        temp = model.predict(info=model.cached)[0][0]
        profit = model.profit(temp)
        print(model.cached[0][0][:3])
        prev_close = model.cached[0][-1][0]
        print(prev_close)
        if profit >= 1.01:
            signals[time_stamp] = 1
            if stocks_hold:
                bought_at.append(prev_close)
        elif profit <= 1:
            for stock in bought_at:
                money_made += prev_close-stock
            bought_at = []
            signals[time_stamp] = -1
        model.start_date = datetime.strptime(model.start_date, "%Y-%m-%d")+relativedelta(days=1)
        model.end_date = datetime.strptime(model.end_date, "%Y-%m-%d")+relativedelta(days=1)
        model.end_date = model.end_date.strftime("%Y-%m-%d")
        model.start_date = model.start_date.strftime("%Y-%m-%d")
    print(money_made)
model = PercentageModel(stock_symbol="GE", information_keys=ImpulseMACD_indicators)
model.start_date="2010-07-04"
model.end_date="2023-08-09"
model.num_days = 10
model.train(epochs=1, use_transfer_learning=False)
test_individual_indepth(model)

