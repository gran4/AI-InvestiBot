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
from threading import Thread
from Models import *

TIME_INTERVAL = 5#86400# number of secs in 24 hours
TICKER = "AAPL"

model = ImpulseMACDModel()
model.load()
#model.get_stock_data_offline()

def run_loop() -> None:
    """This function will attempt to run the loop for the stock bot indefinitely."""
    while True:
        model.update_cached_offline()
        input_data_reshaped = np.reshape(model.cached, (1, 60, model.cached.shape[1]))
        print(model.predict(input_data_reshaped))

        date_object = datetime.strptime(model.start_date, "%Y-%m-%d")
        next_day = date_object + timedelta(days=1)
        model.start_date = next_day.strftime("%Y-%m-%d")

        date_object = datetime.strptime(model.end_date, "%Y-%m-%d")
        next_day = date_object + timedelta(days=1)
        model.end_date = next_day.strftime("%Y-%m-%d")
        time.sleep(TIME_INTERVAL)


def test_accuracy() -> None:
    """This function will attempt to test the accuracy of the model."""
    with open(f'{TICKER}/info.json') as file:
        data = json.load(file)
    i = 200
    together_list = []
    last = 0
    last_predict = 0
    while i < 1000:
        model.update_cached_offline()
        input_data_reshaped = np.reshape(model.cached, (1, 60, model.cached.shape[1]))
        prediction = model.predict(input_data_reshaped) - last_predict
        last = data['Close'][i+1]-last
        last_predict = prediction

        down_together = last_predict[0][0]<0 and last<0
        up_together = last_predict[0][0] >0 and last>0

        together_list.append(down_together or up_together)
        i += 1
        time.sleep(TIME_INTERVAL)
    together_list = [int(a) for a in together_list]

    print(sum(together_list), len(together_list))
    print("percennt correct(in terms of going up or down)", sum(together_list)/len(together_list))

if __name__ == "__main__":
    # Create a new thread
    thread = Thread(target=run_loop)

    # Start the thread
    thread.start()
