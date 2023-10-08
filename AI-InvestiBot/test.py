from models import *

from pandas_market_calendars import get_calendar
import pandas as pd
import numpy as np

from trading_funcs import create_sequences, get_relavant_values, plot
from typing import List


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


def test_indepth(models: List[BaseModel], hold_stocks=False):
    if type(models[0]) == PriceModel:
        raise TypeError('Does not work with price model, sorry')
    processed_data = []
    caches = []
    for model in models:
        cached, data2, _ = get_relavant_values(
            model.stock_symbol, model.information_keys, start_date=model.start_date, end_date=model.end_date
        )
        caches.append(cached)

        temp, temp2 = create_sequences(data2, model.num_days)
        temp_test, expected = model.process_x_y_total(temp, temp2, model.num_days, 0)
        print(temp_test.shape)
        for t in temp_test:
            plot(t[:, 0])
        processed_data.append(temp_test)
    percent_made = 1
    bought_at = []

    signals = {}
    time_stamp = 0

    data = []
    real_data = []
    init = 0
    #return
    while True:
        init += 1
        if init >= 200:
            break
        profits = []
        i = 0
        for model in models:
            expanded_array = np.expand_dims(processed_data[i][init], axis=0)
            temp = model.predict(info=expanded_array)[0][0]
            prev_close = expected[init]
            profit = model.profit(temp, prev_close)
            profits.append((i, profit, prev_close))
            i += 1
            #real_data.append(expected[init])#cached_info['Close'][index+1]/prev_close)
        profits = sorted(profits, key=lambda x: x[1], reverse=True)
        data.append(profits[0][1])
        real_data.append(expected[init])

        if not hold_stocks:
            for stock in bought_at:
                percent_made *= stock[2]/100+1
                print(stock[2])
            bought_at = []
        if profits[0][1] >= 2:
            signals[time_stamp] = 1
            bought_at.append(profits[0])
        elif profit <= .2:
            for stock in bought_at:
                percent_made += stock[1]*stock[2]/100
            bought_at = []
            signals[time_stamp] = -1
    print(percent_made)

models = []
for company in ["AAPL", "HD", "DIS", "GOOG"]:
    model = PercentageModel(stock_symbol=company, information_keys=ImpulseMACD_indicators)
    model.num_days = 10
    model.load()
    model.start_date = "2021-08-05"
    model.end_date = "2023-06-09"
    models.append(model)
test_indepth(models)

