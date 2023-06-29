"""
Name:
    lambda_implementation.py

Purpose:
    This module provides a lambda implementation that is applied to the data
    in the models.

Author:
    Grant Yul Hur

See also:
    Other modules related to running the stock bot -> resource_manager, loop_implementation
"""

from trading_system import *
from Models import (
    BaseModel,
    DayTradeModel,
    MACDModel,
    ImpulseMACDModel,
    EarningsModel,
    ReversalModel,
    BreakoutModel
)
from typing import Dict
import boto3

time_interval = 86400# number of secs in 24 hours


def lambda_handler(event, context) -> Dict:
    """
    This function is the handler for the lambda function. It is called with the provided
    event and context.

    Args:
        event: The event that is passed to the lambda function
        context: The context that is passed to the lambda function
    
    Returns:
        dict: A dictionary that contains the status code and body to indicate success or failure of the request
    """
    update_all()
    print("Success!!!")


    #Optional
    return {
        'statusCode': 200,
        'body': 'Buy order executed sucessfully'
    }

def start_lambda():
    """This function will attempt to create a CloudWatch event rule that will trigger the lambda function."""
    #create Cloud watch event rules
    events_client = boto3.client('events')
    rule_name = 'StockTradingBotRules'
    schedule_expression = f'rate({time_interval} seconds)'

    events_client.client.put_rule(
        Name=rule_name,
        ScheduleExpression=schedule_expression,
        State='Enabled'
    )

    #Add the Lambda function as the for the CloudWatch Events rule
    lambda_client = boto3.client('lambda')
    lambda_func_name = 'StockTradingBot'
    lambda_func_arn = lambda_client.get_function(FunctionName=lambda_func_name)['Configuration']['FunctionArn']
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



if __name__ == "__main__":
    start_lambda()