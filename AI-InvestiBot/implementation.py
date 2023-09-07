"""
Name:
    implementation.py

Purpose:
    This module provides a lambda and loop implementation
    to predict and trade stock/crypto/currency(anything that can be traded).

Author:
    Grant Yul Hur

See also:
    Other modules related to running the stock bot -> resource_manager
"""
import time

from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from typing import Dict, List

from threading import Thread
from pandas_market_calendars import get_calendar

from models import *
from resource_manager import ResourceManager

import numpy as np
#from trading_funcs import company_symbols, is_floats
company_symbols = ["AAPL", "HD", "DIS", "GOOG"]

# API keys from alpaca
YOUR_API_KEY_ID = None
YOUR_SECRET_KEY = None
# API keys from AWS lambda, see boto3 documentation
BUCKET_NAME = 'your_s3_bucket_name'
OBJECT_KEY = 'your_s3_object_key'

# The min predicted profit that every model has to have
# For us to consider buying in. Each has to predict it
# will go in % up more then `PREDICTION_THRESHOLD`
PREDICTION_THRESHOLD = 1

RISK_REWARD_RATIO = 1.01# min profit expected in a stock over a day to hold or buy
TIME_INTERVAL = 86400# number of secs in 24 hours

# if it is lower than `MAX_HOLD_INDEX` and 
# meets all other requirements, hold it
MAX_HOLD_INDEX = 10



if YOUR_API_KEY_ID is None:
    raise ValueError("Set your API key ID")
if YOUR_SECRET_KEY is None:
    raise ValueError("Set your secret key")

RESOURCE_MANAGER = ResourceManager(max_percent=30, api_key=YOUR_API_KEY_ID, secret_key=YOUR_SECRET_KEY)

# for caching for multiple models
def load_models(model_class: BaseModel=PercentageModel, strategys: List[List[str]]=[]):
    """
    Loads all models

    model_classes tells the program what models to use
    """
    models = []
    total_info_keys = []
    for info_keys in strategys:
        total_info_keys += info_keys

    for company in company_symbols:
        temp = []
        models.append(temp)
        for strategy in strategys:
            model = model_class(stock_symbol=company, information_keys=strategy)
            model.load()
            temp.append(model)

    return models, total_info_keys


def set_models_today(models):
    today = date.today().strftime("%Y-%m-%d")
    for company in models:
        for model in company:
            model.end_date = today
    return models


def update_models(models, total_info_keys):
    profits = []

    model = models[0][0]
    end_datetime = datetime.strptime(model.end_date, "%Y-%m-%d")
    nyse = get_calendar('NYSE')
    schedule = nyse.schedule(start_date=model.end_date, end_date=end_datetime+relativedelta(days=2))
    if model.end_date not in schedule.index: # holiday or week ends
        return

    i = 0
    #for company in company_symbols:
    for company_models in models:
        # NOTE: grouping together caches is a small optimization
        temp = company_models[0]
        cached_info = temp.update_cached_info_online()
        cached = temp.indicators_past_num_days(
            model.stock_symbol, temp.end_date,
            total_info_keys, temp.scaler_data,
            cached_info, temp.num_days*2
        )
        predictions = []
        for model in company_models:
            temp = []
            for key in model.information_keys:
                temp.append(cached[key])
            temp_cached = []
            for i in range(model.num_days):
                n = []
                for element in temp:
                    n.append(element[i])
                temp_cached.append(n)
            temp_cached = np.array(temp_cached)
            temp_cached = np.expand_dims(temp_cached, axis=0)

            num_days = model.num_days
            num_windows = num_days#temp_cached.shape[0] - num_days + 1
            # Create a 3D numpy array to store the scaled data
            scaled_data = np.zeros((num_windows, num_days, temp_cached.shape[1], temp_cached.shape[2]))

            for i in range(temp_cached.shape[0]-num_days):
                # Get the data for the current window using the i-window_size approach
                window = temp_cached[i : i + num_days]
                #total 4218, 10 windows, num_days 10, indicators 7

                # Calculate the high and low close prices for the current window
                high_close = np.max(window, axis=0)
                low_close = np.min(window, axis=0)

                # Avoid division by zero if high_close and low_close are equal
                scale_denominator = np.where(high_close == low_close, 1, high_close - low_close)

                # Scale each column using broadcasting
                scaled_window = (window - low_close) / scale_denominator
                # Store the scaled window in the 3D array
                scaled_data[i] = scaled_window

            model.cached = scaled_data
            temp = model.predict(info=model.cached)[0][0]
            prev_close = float(model.cached[-1][0][-1][0])
            profit = model.profit(temp, prev_close)
            predictions.append(temp)
            #input_data_reshaped = np.reshape(model.cached, (1, 60, model.cached.shape[1]))
        profits.append(predictions)
        i += len(models)

    processed_profits = []
    for profit in profits:
        print(profit)
        # If it is good enough, it can possibly be bought even if one model is lower then the PREDICTION_THRESHOLD
        filtered_profit = [0 if model_prediction < PREDICTION_THRESHOLD else model_prediction for model_prediction in profit]
        average_profit = sum(filtered_profit) / len(filtered_profit)
        processed_profits.append(average_profit)
    model_weights = list(zip(models, processed_profits))
    sorted_models = sorted(model_weights, key=lambda x: x[1], reverse=True)
    i = 0
    print(sorted_models)
    for model, profit in sorted_models:
        if RESOURCE_MANAGER.is_in_portfolio(model[0].stock_symbol) and i<MAX_HOLD_INDEX:
            RESOURCE_MANAGER.sell()
        if profit < RISK_REWARD_RATIO:
            break
        RESOURCE_MANAGER.buy(model[0].stock_symbol)
        i += 1


def run_loop(models, total_info_keys) -> None:
    """Runs the stock bot in a loop"""
    day = 0
    while True:
        day += 1
        print("day: ", day)
        models = set_models_today(models)
        for company in models:
            for model in company:
                model.num_days = 10
                model.end_date = "2023-08-31"
        update_models(models, total_info_keys)
        time.sleep(TIME_INTERVAL)





#vvvvvvvvvvv---Lambda----Painless-version----RECOMENDED-vvvvvvvvv#

import boto3
import json


def lambda_handler(event, context) -> Dict:
    """
    This function is the handler for the lambda function.
    It is called with the provided event and context.

    Args:
        event: The event that is passed to the lambda function
        context: The context that is passed to the lambda function
    
    Returns:
        dict: A dictionary that contains the status code and body
        to indicate success or failure of the request
    """
    # Read the current state from S3
    current_state = read_state_from_s3()
    model = current_state.get('model', {})
    total_info_keys = current_state.get('total_info_keys', {})

    models = set_models_today(models)
    update_models(models, total_info_keys)

    # Save the updated state to S3
    save_state_to_s3(model, total_info_keys)

    #Optional
    return {
        'statusCode': 200,
        'body': 'Buy order executed sucessfully'
    }

def start_lambda(model, total_info_keys):
    """This function will attempt to create a CloudWatch event
    rule that will trigger the lambda function."""
    #create Cloud watch event rules
    events_client = boto3.client('events')
    rule_name = 'StockTradingBotRules'
    schedule_expression = f'rate({TIME_INTERVAL} seconds)'

    events_client.client.put_rule(
        Name=rule_name,
        ScheduleExpression=schedule_expression,
        State='Enabled'
    )

    #Add the Lambda function as the for the CloudWatch Events rule
    lambda_client = boto3.client('lambda')
    lambda_func_name = 'StockTradingBot'
    lambda_func = lambda_client.get_function(FunctionName=lambda_func_name)
    lambda_func_arn = lambda_func['Configuration']['FunctionArn']
    targets = [
        {
            'ID': '1',
            'Arn': lambda_func_arn
        },
    ]
    events_client.put_targets(
        Rule = rule_name,
        Targets=targets
    )

    # Save the updated state to S3
    save_state_to_s3(model, total_info_keys)

def read_state_from_s3():
    s3 = boto3.resource('s3')
    try:
        obj = s3.Object(BUCKET_NAME, OBJECT_KEY).get()
        return json.loads(obj['Body'].read())
    except s3.meta.client.exceptions.NoSuchKey:
        # Return an empty state if the key does not exist
        return {}

def save_state_to_s3(model, total_info_keys):
    s3 = boto3.resource('s3')
    state = {
        'model': model,
        'total_info_keys': total_info_keys
    }
    s3.Object(BUCKET_NAME, OBJECT_KEY).put(Body=json.dumps(state))


if __name__ == "__main__":
    """NOTE: runs loop ONLY unless you change it"""
    # Create a new thread
    models, total_info_keys = load_models(strategys=[ImpulseMACD_indicators])
    thread = Thread(target=run_loop, args=(models, total_info_keys))

    # Start the thread
    thread.start()
