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
import json
import boto3
import warnings

from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional
from threading import Thread
from pandas_market_calendars import get_calendar
from resource_manager import ResourceManager
from models import *

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

try: # import keys from config file
    with open("secrets.config","rb") as f:
        secrets = json.load(f)
except FileNotFoundError:
    print("missing 'secrets.config' file, look at the 'secrets_example.config' file for an example of what to do")
    raise

try:
    # API keys from alpaca
    YOUR_API_KEY_ID = secrets["alpaca_api_key"]
    YOUR_SECRET_KEY = secrets["alpaca_secret_key"]
    # API keys from AWS lambda, see boto3 documentation
    BUCKET_NAME = secrets["aws_bucket_name"]
    OBJECT_KEY = secrets["aws_object_key"]
except KeyError:
    print("missing keys in 'secrets.config' file.")
    raise

if YOUR_API_KEY_ID is None or YOUR_API_KEY_ID=="":
    raise ValueError("Set your API key ID from alpaca")
if YOUR_SECRET_KEY is None or YOUR_SECRET_KEY=="":
    raise ValueError("Set your secret key from alpaca")
if BUCKET_NAME is None or BUCKET_NAME=="":
    warnings.warn("Set your BUCKET_NAME from AWS", category=RuntimeWarning)
if OBJECT_KEY is None or OBJECT_KEY=="":
    warnings.warn("Set your OBJECT_KEY from AWS", category=RuntimeWarning)

# The min predicted profit that every model has to have
# For us to consider buying in. Each has to predict it
# will go in % up more then `PREDICTION_THRESHOLD`
PREDICTION_THRESHOLD = 1

RISK_REWARD_RATIO = 1.01# min profit expected in a stock over a day to hold or buy
TIME_INTERVAL = 86400# number of secs in 24 hours

# if it is lower than `MAX_HOLD_INDEX` and 
# meets all other requirements, hold it
MAX_HOLD_INDEX = 3

# for caching for multiple models
def load_models(model_class: BaseModel=PercentageModel, strategys: List[List[str]]=[], names: List[str]=[], company_symbols: List[str]=["AAPL", "GOOG", "AMZN", "META", 'MSFT', 'TSLA', 'V', 'JPM', 'WMT', 'DIS']):
    """
    Loads all models

    model_classes tells the program what models to use
    """
    if len(names) == 0: # no names given when loading, just use base names
        names = [None for i in range(len(strategys))]

    models = []
    total_info_keys = []
    for info_keys in strategys:
        total_info_keys += info_keys

    for company in company_symbols:
        temp = []
        models.append(temp)
        for i in range(len(strategys)):
            model = model_class(stock_symbol=company, information_keys=strategys[i])
            model.num_days = 14
            model.load(name=names[i])
            temp.append(model)
    return models, total_info_keys


def set_models_today(models):
    today = date.today().strftime("%Y-%m-%d")

    current_date = datetime.now()
    ten_days_ago = current_date# - timedelta(days=models[0][0].num_days*2 + 2)
    ten_days_ago = ten_days_ago.strftime("%Y-%m-%d")
    for company in models:
        for model in company:
            model.start_date = ten_days_ago
            model.end_date = today
            print(today)
    return models

def accuracy(close_vals, predicted, days):
    temp = close_vals.pct_change() * 100
    import pandas as pd
    df = pd.DataFrame({'Predicted': predicted.flatten(), 'Actual': temp.tail(days)[:-1]})
    # Create a column indicating whether the predicted direction matches the actual direction
    df['DirectionMatch'] = np.sign(df['Predicted']) == np.sign(df['Actual'])
    accuracy = (df['DirectionMatch'].sum() / len(df)) * 100
    # Display accuracy
    print(f'Accuracy: {accuracy:.2f}%')

def update_models(models, total_info_keys, manager: ResourceManager):
    profits = []

    model = models[0][0]
    end_datetime = datetime.strptime(model.end_date, "%Y-%m-%d")
    nyse = get_calendar('NYSE')
    schedule = nyse.schedule(start_date=model.end_date, end_date=end_datetime+relativedelta(days=2))
    if model.end_date not in schedule.index: # holiday or week ends
        return

    i = 0
    for company_models in models:
        # NOTE: grouping together caches is a small optimization
        temp = company_models[0]
        cached_info = temp.update_cached_info_online()
        cached = temp.indicators_past_num_days(
            model.stock_symbol, temp.end_date,
            total_info_keys, temp.scaler_data,
            cached_info, temp.num_days
        )
        predictions = []
        for model in company_models:
            processed_data = model.process_cached(cached)
            temp = model.predict(info=processed_data)[0][-1]

            prev_close = float(cached['Close'][-1])
            profit = model.profit(temp, prev_close)
            predictions.append(temp)
        profits.append(predictions)
        i += len(models)

    processed_profits = []
    for profit in profits:
        # If it is good enough, it can possibly be bought even if one model is lower then the PREDICTION_THRESHOLD
        filtered_profit = [0 if model_prediction < PREDICTION_THRESHOLD else model_prediction for model_prediction in profit]
        average_profit = sum(filtered_profit) / len(filtered_profit)
        processed_profits.append(average_profit)
    model_weights = list(zip(models, processed_profits))
    sorted_models = sorted(model_weights, key=lambda x: x[1], reverse=True)
    i = 0

    sellable_amounts = manager.get_sellable_amounts()
    # Get a list of your positions and open orders
    for model, profit in sorted_models:
        symbol = model[0].stock_symbol
        if manager.is_in_portfolio(symbol) and i<MAX_HOLD_INDEX and sellable_amounts[symbol] > 0:
            manager.sell(sellable_amounts[symbol], symbol)
        if profit < RISK_REWARD_RATIO:
            break
        manager.buy(symbol)
        i += 1


def run_loop(models, total_info_keys, manager: Optional[ResourceManager]=None) -> None:
    """Runs the stock bot in a loop"""
    if manager is None:
        manager = ResourceManager(max_percent=30, api_key=YOUR_API_KEY_ID, secret_key=YOUR_SECRET_KEY)

    day = 0
    while True:
        day += 1
        print("day: ", day)
        models = set_models_today(models)
        update_models(models, total_info_keys, manager)
        time.sleep(TIME_INTERVAL)





#vvvvvvvvvvv---Lambda----Painless-version----RECOMENDED-vvvvvvvvv#
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
    manager = current_state.get('manager', {})

    models = set_models_today(models)
    update_models(models, total_info_keys, manager)

    # Save the updated state to S3
    save_state_to_s3(model, total_info_keys, manager)

    #Optional
    return {
        'statusCode': 200,
        'body': 'Buy order executed sucessfully'
    }

def start_lambda(model, total_info_keys, manager: Optional[ResourceManager]=None):
    """This function will attempt to create a CloudWatch event
    rule that will trigger the lambda function."""
    if manager is None:
        manager = ResourceManager(max_percent=30, api_key=YOUR_API_KEY_ID, secret_key=YOUR_SECRET_KEY)
    
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
    save_state_to_s3(model, total_info_keys, manager)

def read_state_from_s3():
    s3 = boto3.resource('s3')
    obj = s3.Object(BUCKET_NAME, OBJECT_KEY).get()
    return json.loads(obj['Body'].read())

def save_state_to_s3(model, total_info_keys, manager: ResourceManager):
    s3 = boto3.resource('s3')
    state = {
        'model': model,
        'total_info_keys': total_info_keys,
        'manager': manager
    }
    s3.Object(BUCKET_NAME, OBJECT_KEY).put(Body=json.dumps(state))


if __name__ == "__main__":
    # NOTE: runs loop ONLY, unless you change it
    # Create a new thread
    models, total_info_keys = load_models(strategys=[break_out_indicators])
    thread = Thread(target=run_loop, args=(models, total_info_keys))

    # Start the thread
    thread.start()

trading_calendar = get_calendar('XNYS')
def save_data_for_predictions(company_models, start_date):
    predictions = []
    for model in company_models:
        predictions.append([])
    initial_date = pd.Timestamp(start_date, tz='America/Los_Angeles')
    initial_date = initial_date.tz_convert('UTC')
    new_date = trading_calendar.valid_days(start_date=initial_date, end_date=initial_date + pd.DateOffset(days=14))[-1]
    # Define the comparison date (2023-10-11 in this case)
    comparison_date = pd.Timestamp("2023-10-11", tz='America/Los_Angeles')
    # Check if the new date is past the comparison date
    assert new_date.tzinfo == initial_date.tzinfo
    
    first_model = company_models[0]
    cached_info = first_model.update_cached_info_online()
    cached = first_model.indicators_past_num_days(
        model.stock_symbol, start_date,
        total_info_keys, first_model.scaler_data,
        cached_info, first_model.num_days
    )
    while new_date < comparison_date:
        i = 0
        for model in company_models:
            processed_data = model.process_cached(cached)
            temp = model.predict(info=processed_data).flatten()
            temp = temp[::-1].tolist()
            predictions[i] += temp
            i += 1
        new_date = trading_calendar.valid_days(start_date=new_date, end_date=new_date + pd.DateOffset(days=14))[-1]
    return predictions
models, total_info_keys = load_models(strategys=[break_out_indicators, ImpulseMACD_indicators, Reversal_indicators, RSI_indicators, super_trends_indicators], names=['breakout', 'ImpulseMACD', 'Reversal', 'RSI', 'supertrends'])
data = {}
for company_models in models:
    print(type(company_models[0]))
    data[company_models[0].stock_symbol] = save_data_for_predictions(company_models, "2015-01-01")
with open(f"Stocks/data_for_decision_tree.json", "w") as json_file:
    json.dump(data, json_file)



