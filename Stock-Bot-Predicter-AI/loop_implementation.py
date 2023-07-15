"""
Name:
    Loop_implementation.py

Purpose:
    This module provides a loop implementation that is applied to the data
    in the models.

Author:
    Grant Yul Hur

See also:
    Other modules related to running the stock bot -> resource_manager, lambda_implementation
"""
import time
from datetime import datetime, timedelta, date

from threading import Thread
from pandas_market_calendars import get_calendar
import numpy as np

from models import *
from resource_manager import ResourceManager
from trading_funcs import company_symbols, is_floats

YOUR_API_KEY_ID = None
YOUR_SECRET_KEY = None
# btw 10 and 25
# btw 15 and 20
models = []
for i in range(30, 31, 5):
    model = RSIModel()
    model.train(nurons=i, epochs=2000)
    #model.load()
    model.t = i
    models.append(model)
for model in models:
    print(model.t)
    model.test()
raise ValueError('de')
# The min predicted profit that every model has to have
# For us to consider buying in. Each has to predict it
# will go in % up more then `PREDICTION_THRESHOLD`
PREDICTION_THRESHOLD = 1

RISK_REWARD_RATIO = 1.01# min profit expected in a stock over a day to hold or buy
TIME_INTERVAL = 0# 86400# number of secs in 24 hours

# if it is lower than `MAX_HOLD_INDEX` and 
# meets all other requirements, hold it
MAX_HOLD_INDEX = 10

models = []
model_classes = [ImpulseMACDModel, EarningsModel, RSIModel]

# for caching for multiple models
total_info_keys = []
for model in model_classes:
    total_info_keys += model().information_keys

for company in company_symbols:
    for model_class in model_classes:
        model = model_class(stock_symbol=company)
        model.load()
        models.append(model)


YOUR_API_KEY_ID = "PKJWNCBFPYBEFZ9GLA5B"
YOUR_SECRET_KEY = "Jl2ujDJ6AsrK8Ytu1DqBuuxcZb6hh6RbiKjzLYup"
if YOUR_API_KEY_ID is None:
    raise ValueError("Set your API key ID")
if YOUR_SECRET_KEY is None:
    raise ValueError("Set your secret key")

RESOURCE_MANAGER = ResourceManager(max_percent=30, api_key=YOUR_API_KEY_ID, secret_key=YOUR_SECRET_KEY)

def run_loop() -> None:
    """Runs the stock bot in a loop"""
    today = date.today()
    for model in models:
        model.end_date = today.strftime("%Y-%m-%d")
    while True:
        skip = False
        profits = []

        model = models[0]
        end_datetime = datetime.strptime(model.end_date, "%Y-%m-%d")
        nyse = get_calendar('NYSE')
        schedule = nyse.schedule(start_date=model.end_date, end_date=end_datetime+timedelta(days=2))
        if model.end_date not in schedule.index: # holiday or week ends
            time.sleep(TIME_INTERVAL)
            continue

        temp = models[0]
        cached_info = temp.update_cached_info_online()
        cached = temp.indicators_past_num_days(
            company, temp.end_date,
            total_info_keys, temp.scaler_data,
            cached_info, temp.num_days
        )
        for model in models:
            model.cached_info = cached_info
            cached = [cached[key] for key in model.information_keys if is_floats(cached[key])]
            model.cached = np.transpose(cached)
            info = model.update_cached_online()
            if info is None:
                skip = True
                break
            elif len(info) == 0:
                raise RuntimeError("My best guess is that `end_date` is past today")  

            #input_data_reshaped = np.reshape(model.cached, (1, 60, model.cached.shape[1]))
            prev_close = float(info[0][-1][0])
            temp = model.predict(info=info)[0][0]
            profit = float(temp/prev_close)
            print(model.company_symbol, profit)
            profits.append(profit)
        if skip:
            time.sleep(TIME_INTERVAL)
            continue

        processed_profits = []
        for profit in profits:
            # If it is good enough, it can possibly be bought even if one model is lower then the PREDICTION_THRESHOLD
            filtered_profit = [0 if model_prediction < PREDICTION_THRESHOLD else model_prediction for model_prediction in profit]
            average_profit = sum(filtered_profit) / len(filtered_profit)
            processed_profits.append(average_profit)
        model_weights = list(zip(models, processed_profits))
        sorted_models = sorted(model_weights, key=lambda x: x[1], reverse=True)
        i = 0
        for model, profit in sorted_models:
            if RESOURCE_MANAGER.is_in_portfolio(model.stock_symbol) and i<MAX_HOLD_INDEX:
                RESOURCE_MANAGER.sell()
            if profit < RISK_REWARD_RATIO:
                break
            amount = RESOURCE_MANAGER.check(model.stock_symbol)
            if amount == 0:
                break
            RESOURCE_MANAGER.buy(model.stock_symbol, amount=amount)
            i += 1
        time.sleep(TIME_INTERVAL)


if __name__ == "__main__":
    # Create a new thread
    thread = Thread(target=run_loop)

    # Start the thread
    thread.start()
