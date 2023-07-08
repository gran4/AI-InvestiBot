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
import numpy as np

from models import DayTradeModel
from resource_manager import ResourceManager

YOUR_API_KEY_ID = None
YOUR_SECRET_KEY = None
RISK_REWARD_RATIO = 1.01#min profit expected per day to hold or buy
TIME_INTERVAL = 0#86400# number of secs in 24 hours
MAX_HOLD_INDEX = 10
# if it is lower than `MAX_HOLD_INDEX` and more then
# `RISK_TO_REWARD_RATIO`, hold it

models = []
model = DayTradeModel()
model.load()
models.append(model)

YOUR_API_KEY_ID = "CK12XB0M57U33N2RBD9L"
YOUR_SECRET_KEY = "LDUfOeFq2SxPRAFMCSSfj7vAQ49mw7CmXAvg4GXZ"
if YOUR_API_KEY_ID is None:
    raise ValueError("Set your API key ID")
if YOUR_SECRET_KEY is None:
    raise ValueError("Set your secret key")

RESOURCE_MANAGER = ResourceManager(api_key=YOUR_API_KEY_ID, secret_key=YOUR_SECRET_KEY)

def run_loop() -> None:
    """Runs the stock bot in a loop"""
    while True:
        weights = []
        for model in models:
            if model.get_info_today() is None:
                raise RuntimeError("`end_date` is past today")  

            input_data_reshaped = np.reshape(model.cached, (1, 60, model.cached.shape[1]))
            prev_close = model.cached[-1][0]
            temp = model.predict(info=input_data_reshaped)

            weight = temp/prev_close
            weights.append(weight)
        i = 0
        for weight, model in zip(weights, models):
            if RESOURCE_MANAGER.is_in_portfolio(model.symbol) and i<MAX_HOLD_INDEX:
                RESOURCE_MANAGER.sell()
            if weight > RISK_REWARD_RATIO:
                break
            RESOURCE_MANAGER.buy(model.symbol)
            i += 1


if __name__ == "__main__":
    # Create a new thread
    thread = Thread(target=run_loop)

    # Start the thread
    thread.start()
