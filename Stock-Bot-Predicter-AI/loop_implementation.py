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
from threading import Thread
from datetime import datetime, timedelta

import numpy as np
import alpaca_trade_api as tradeapi, REST

from models import DayTradeModel
from resource_manager import ResourceManager

YOUR_API_KEY_ID = None
YOUR_SECRET_KEY = None
RISK_REWARD_RATIO = 1.01#min profit expected
models = []

YOUR_API_KEY_ID = None
YOUR_SECRET_KEY = None
if YOUR_API_KEY_ID is None:
    raise ValueError("Set your API key ID")
if YOUR_SECRET_KEY is None:
    raise ValueError("Set your secret key")
api = tradeapi.REST(YOUR_API_KEY_ID, YOUR_SECRET_KEY, base_url='https://paper-api.alpaca.markets')
account = api.get_account()

TIME_INTERVAL = 0#86400# number of secs in 24 hours
TICKER = "AAPL"

model = DayTradeModel()
model.load()
models.append(model)
#model.get_stock_data_offline()

RESOURCE_MANAGER = ResourceManager

def run_loop() -> None:
    """Runs the stock bot in a loop"""
    while True:
        if model.get_info_today() is None:
            raise RuntimeError("`end_date` is past today")  

        input_data_reshaped = np.reshape(model.cached, (1, 60, model.cached.shape[1]))
        prev_close = model.cached[-1][0]
        temp = model.predict(info=input_data_reshaped)

        weight = temp/prev_close
        if weight > RISK_REWARD_RATIO:
            pass


if __name__ == "__main__":
    # Create a new thread
    thread = Thread(target=run_loop)

    # Start the thread
    thread.start()
