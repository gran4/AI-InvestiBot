"""
Name:
    trading_funcs.py

Description:
    This module provides functions to use in Models
    during getting data, training, and predicting.

    Get earnings, sequences, scaling, and relevant Values.

Author:
    Grant Yul Hur

See also:
    Similarly related modules -> E.g Models.py, 
"""

import json

from typing import List, Tuple
from pandas_market_calendars import get_calendar
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


#values that do not go from day to day
#EX: earnings comeout every quarter
excluded_values = (
    "earnings dates",
    "earnings diff"
)


company_symbols = (
    "AAPL"
)


def create_sequences(data: np.array, num_days: int) -> Tuple[np.array, np.array]:
    """
    The purpose of this function is to create sequences and labels which are implemented
    into the model during fitting. This is done by iterating through the data and appending
    the data to the sequences and labels list.

    Args:
        data (np.array): The data which is used to create the sequences and labels
        num_days (int): The number of days which is used to create the sequences

    Returns:
        Tuple[np.array[sequences (list)], np.array[labels (list)]]: The sequences and labels which are used in the model
        
    Sequences are the input and Labels are the output
    """

    sequences = [] # What inputs look like
    labels = [] # What output looks like
    for i in range(num_days, len(data)):
        sequences.append(data[i-num_days:i])
        labels.append(data[i, 0])
    return np.array(sequences), np.array(labels)


def process_earnings(dates: List, diffs: List, start_date: str, end_date: str):
    """
    The purpose of this function is to process the earnings between the start and end date range
    and fill in the 0s for dates without an earnings report. 

    Args:
        dates (List): The dates which are used to get the earnings
        diffs (List): The earnings which are used to get the earnings
        start_date (str): The minimum start date which is used to get the earnings
        end_date (str): The maximum end date which is used to get the earnings
    
    Returns:
        dates (List): The dates which are used to get the earnings
        diffs (List): The earnings which are used to get the earnings
    """
    #_________________deletes earnings before start and after end______________________#
    start = 0
    end = len(dates)
    for day in dates:
        if day < start_date:
            start += 1
        elif day > end_date:
            end += 1
    dates = dates[start:end]
    diffs = diffs[start:end]


    #_________________Fill Data out with 0s______________________#
    filled_dates = []
    filled_earnings = []

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Fill out the list to match the start and end date
    dates = [datetime.strptime(date_str, "%b %d, %Y") for date_str in dates]
    current_date = start_date
    while current_date <= end_date:
        filled_dates.append(current_date)
        if current_date in dates:
            existing_index = dates.index(current_date)
            filled_earnings.append(diffs[existing_index])
        else:
            filled_earnings.append(0)
        current_date += timedelta(days=1)
    return dates, diffs


def process_flips(ema12: np.array, ema26: np.array) -> List:
    """
    The purpose of this function is to process the flips between the 12-day ema and 26-day ema. It
    is primarily used for the MACD indicator. This is done by iterating through the ema12 and ema26
    and appending the True or False value to the temp list. 

    Args:
        ema12 (np.array): The 12-day ema which is used to get the flips
        ema26 (np.array): The 26-day ema which is used to get the flips

    Returns:
        List[map(int, temp)]: The list of flips between the 12-day ema and 26-day ema
    """
    temp = []
    shortmore = None
    for short, mid in zip(ema12, ema26):
        if shortmore is None:
            shortmore = short>mid
        elif shortmore and short<mid:
            temp.append(True)
            shortmore = False
            continue
        elif not shortmore and short>mid:
            temp.append(True)
            shortmore = True
            continue
        temp.append(False)
    return list(map(int, temp))


def get_relavant_values(start_date: str, end_date: str, stock_symbol: str, information_keys: List[str]) -> np.array:
    """
    The purpose of this function is to get the relevant values between the start and end date range
    as well as the corrected dates.

    Args:
        start_date (str): The minimum start date which is used to get the relevant values
        end_date (str): The maximum end date which is used to get the relevant values
        stock_symbol (str): The stock symbol which is used to get the relevant values
        information_keys (List[str]): The information keys which are used to get the relevant values

    Returns:
        (np.array) [filtered (NDArray[Any]), start_date (str), end_date (str)]: The relevant filtered values, start date, and end date
    """    
    #_________________Check if start or end is holiday______________________#
    nyse = get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    #_________________Change if it is a holiday______________________#
    start_date = pd.to_datetime(start_date).date()
    if start_date not in schedule.index:
        # Find the next trading day
        next_trading_day = nyse.valid_days(start_date=start_date, end_date=end_date)[0]
        start_date = next_trading_day.date().strftime('%Y-%m-%d')

    end_date = pd.to_datetime(end_date).date()
    if end_date not in schedule.index:
        end_date = schedule.index[-1].date().strftime('%Y-%m-%d')


    #_________________Load info______________________#
    with open(f'{stock_symbol}/info.json') as file:
        other_vals = json.load(file)

    #_________________Make Data fit between start and end date______________________#
    if start_date in other_vals['Dates']:
        i = other_vals['Dates'].index(start_date)
        other_vals['Dates'] = other_vals['Dates'][i:]
        for key in information_keys:
            if key in excluded_values:
                continue
            other_vals[key] = other_vals[key][i:]
    else:
        raise ValueError(f"Run getInfo.py with start date before {start_date}\n and end date after {start_date}")
    if end_date in other_vals['Dates']:
        i = other_vals['Dates'].index(end_date)
        other_vals['Dates'] = other_vals['Dates'][:i]
        for key in information_keys:
            if key in excluded_values:
                continue
            other_vals[key] = other_vals[key][:i]
    else:
        raise ValueError(f"Run getInfo.py with start date before {end_date}\n and end date after {end_date}")

    #_________________Process earnings______________________#
    if "earnings diff" in information_keys:
        dates = other_vals['earnings dates']
        diffs = other_vals['earnings diff']

        dates, diffs = process_earnings(dates, diffs, start_date, end_date)
        other_vals['earnings dates'] = dates
        other_vals['earnings diff'] = diffs

    #_________________Scales Data______________________#
    for key in information_keys:
        if pd.api.types.is_numeric_dtype(other_vals[key]):
            continue
        other_vals[key] = np.array(other_vals[key])

        # Calculate the minimum and maximum values
        min_value = np.min(other_vals[key])
        max_value = np.max(other_vals[key])

        if max_value - min_value != 0:
            # Scale the data
            other_vals[key] = (other_vals[key] - min_value) / (max_value - min_value)

    # Convert the dictionary of lists to a NumPy array
    filtered = [other_vals[key] for key in information_keys if key not in excluded_values]

    filtered = np.transpose(filtered)
    return filtered, start_date, end_date


def scale(num: float, data: List) -> float:
    """
    Scale the number between the minimum and maximum values of the data

    Args:
        num (float): The number which is to be scaled
        data (List): The data which is used to scale the number
    
    Returns:
        (float) The scaled number
    """
    low, high = min(data), max(data)
    return (num - low) / (high - low)
    
