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

from datetime import datetime, timedelta, date
from typing import Optional, List, Dict

from threading import Thread
from pandas_market_calendars import get_calendar

from models import *
from resource_manager import ResourceManager
from trading_funcs import company_symbols, is_floats

YOUR_API_KEY_ID = None
YOUR_SECRET_KEY = None

# The min predicted profit that every model has to have
# For us to consider buying in. Each has to predict it
# will go in % up more then `PREDICTION_THRESHOLD`
PREDICTION_THRESHOLD = 1

RISK_REWARD_RATIO = 1.01# min profit expected in a stock over a day to hold or buy
TIME_INTERVAL = 0# 86400# number of secs in 24 hours

# if it is lower than `MAX_HOLD_INDEX` and 
# meets all other requirements, hold it
MAX_HOLD_INDEX = 10


YOUR_API_KEY_ID = "PKJWNCBFPYBEFZ9GLA5B"
YOUR_SECRET_KEY = "Jl2ujDJ6AsrK8Ytu1DqBuuxcZb6hh6RbiKjzLYup"
if YOUR_API_KEY_ID is None:
    raise ValueError("Set your API key ID")
if YOUR_SECRET_KEY is None:
    raise ValueError("Set your secret key")

RESOURCE_MANAGER = ResourceManager(max_percent=30, api_key=YOUR_API_KEY_ID, secret_key=YOUR_SECRET_KEY)

# for caching for multiple models
def load_models(model_classes=[ImpulseMACDModel]):
    """
    Loads all models

    model_classes tells the program what models to use
    """
    models = []
    total_info_keys = []
    for model in model_classes:
        total_info_keys += model().information_keys

    for company in company_symbols:
        for model_class in model_classes:
            model = model_class(stock_symbol=company)
            model.load()
            models.append(model)

    return models, total_info_keys


def set_models_today(models):
    today = date.today().strftime("%Y-%m-%d")
    for model in models:
        model.end_date = today
    return models


def update_models(models, total_info_keys):
    profits = []

    model = models[0]
    end_datetime = datetime.strptime(model.end_date, "%Y-%m-%d")
    nyse = get_calendar('NYSE')
    schedule = nyse.schedule(start_date=model.end_date, end_date=end_datetime+timedelta(days=2))
    if model.end_date not in schedule.index: # holiday or week ends
        return

    i = 0
    for company in company_symbols:
        # NOTE: grouping together caches is a small optimization
        temp = models[0]
        cached_info = temp.update_cached_info_online()
        cached = temp.indicators_past_num_days(
            company, temp.end_date,
            total_info_keys, temp.scaler_data,
            cached_info, temp.num_days
        )
        for model_index in range(i):
            model = models[model_index]
            model.cached = cached

            #input_data_reshaped = np.reshape(model.cached, (1, 60, model.cached.shape[1]))
            prev_close = float(model.cached[0][-1][0])
            temp = model.predict(info=model.cached)[0][0]
            profit = float(temp/prev_close)
            profits.append(profit)
        i += len(model_classes)

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


def run_loop(models, total_info_keys) -> None:
    """Runs the stock bot in a loop"""
    while True:
        print("DEDNJDENDEJNDEJN")
        models = set_models_today(models)
        update_models(models, total_info_keys)
        time.sleep(TIME_INTERVAL)





#vvvvvvvvvvv---Lambda----Painless-version----RECOMENDED-vvvvvvvvv#

import boto3
import json
BUCKET_NAME = 'your_s3_bucket_name'
OBJECT_KEY = 'your_s3_object_key'


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
    model, total_info_keys = load_models(model_classes=[ImpulseMACDModel, EarningsModel, RSIModel])
    thread = Thread(target=run_loop, args=(model, total_info_keys))

    # Start the thread
    thread.start()