from models import *

from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import json

from pandas_market_calendars import get_calendar
import pandas as pd
import numpy as np

from trading_funcs import create_sequences, get_relavant_values
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
    cached_infos = []
    processed_data = []
    for model in models:

        with open(f"Stocks/{model.stock_symbol}/info.json", 'r') as file:
            cached_info = json.load(file)

        cached_info, data2, _ = get_relavant_values(
            model.stock_symbol, model.information_keys, start_date=model.start_date, end_date=model.end_date
        )
        cached_infos.append(cached_info)

        temp, temp2 = create_sequences(data2, model.num_days)
        temp_test, expected = model.process_x_y_total(temp, temp2, model.num_days, 0)
        processed_data.append(temp_test)
    money_made = 0
    bought_at = []

    signals = {}
    time_stamp = 0

    data = []
    real_data = []
    index = len(cached_infos[0]['Dates'])-10-model.num_days
    init = 0
    while True:
        init += 1
        if init >= index:
            break
        profits = []
        i = 0
        for model in models:
            expanded_array = np.expand_dims(processed_data[i][init], axis=0)
            temp = model.predict(info=expanded_array)[0][0]
            profit = model.profit(temp)
            prev_close = cached_infos[i]['Close'][init]
            profits.append((i, profit, prev_close))
            print(init)
            i += 1
            #real_data.append(expected[init])#cached_info['Close'][index+1]/prev_close)
        profits = sorted(profits, key=lambda x: x[1])

        if not hold_stocks:
            for stock in bought_at:
                money_made += cached_infos[stock[0]]['Close'][init]-stock[2]
            bought_at = []
        if profit >= 2:
            signals[time_stamp] = 1
            bought_at.append(profits[0])
        elif profit <= .2:
            for stock in bought_at:
                money_made += cached_infos[stock[0]]['Close'][init]-stock[2]
            bought_at = []
            signals[time_stamp] = -1
    for i in range(len(real_data)):
        num = real_data[i]
        #num -= 1
        num /= 20
        real_data[i] = num

    return
    import matplotlib.pyplot as plt
    days_train = [i for i in range(len(data))]
    plt.figure(figsize=(18, 6))

    predicted_test = plt.plot(days_train, data, label='Predicted Test')
    real_test = plt.plot(days_train, real_data, label='Predicted Test')
    plt.title(f'TITLE')
    plt.xlabel("X")
    plt.ylabel("Y")

    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(7))

    plt.legend(
        [predicted_test[0], real_test],#[real_data, actual_test[0], actual_train],
        ['Data', "REAL"]#['Real Data', 'Actual Test', 'Actual Train']
    )
    plt.show()
    print(money_made)

models = []
for company in ["AAPL", "HD", "DIS", "GOOG"]:
    model = PercentageModel(stock_symbol=company, information_keys=ImpulseMACD_indicators)
    model.start_date = "2020-07-04"
    model.end_date = "2023-08-09"
    model.num_days = 10
    model.load()
    models.append(model)
test_indepth(models)

