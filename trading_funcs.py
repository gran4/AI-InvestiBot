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
    Similarly related modules involving use of the trading functions -> E.g Models.py, getInfo.py
"""

import json

from typing import List, Tuple, Dict
from datetime import datetime, timedelta

from pandas_market_calendars import get_calendar

import numpy as np
import pandas as pd

__all__ = (
    'excluded_values',
    'company_symbols',
    'create_sequences',
    'process_earnings',
    'process_flips',
    'get_relavant_values',
    'get_scaler',
    'supertrends',
    'kumo_cloud'
)


#values that do not go from day to day
#EX: earnings comeout every quarter
excluded_values = (
    "Dates",
    "earnings dates",
    "earnings diff"
)


company_symbols = (
    "AAPL",
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
        tuple: A tuple containing two NumPy arrays.
            - sequences (np.ndarray): An array representing the input of the model
            - label (np.ndarray): An array representing the output of the model
    """
    sequences = [] # What inputs look like
    labels = [] # What output looks like
    for i in range(num_days, len(data)):
        sequences.append(data[i-num_days:i])
        labels.append(data[i, 0])
    return np.array(sequences), np.array(labels)


def process_earnings(dates: List, diffs: List, start_date: str,
                     end_date: str) -> Tuple[List[str], List[float]]:
    """
    The purpose of this function is to process the earnings between the start and
    end date range, and fill in the 0s for dates without an earnings report. 

    Args:
        dates (list): The dates which are used to get the earnings
        diffs (list): The earnings which are used to get the earnings
        start_date (str): The start date which is used to get the earnings
        end_date (str): The end date which is used to get the earnings
    
    Returns:
        tuple: A tuple containing two Lists.
            - dates (list): The dates which are used to align the earnings
            - diffs (list) The earnings differences bettween the expected
            and actual earnings per share
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


def process_flips(ema12: pd.Series, ema26: pd.Series) -> pd.Series:
    """
    The purpose of this function is to return a list of the 12 and 26 ema flips. It
    is primarily used for the MACD Model.

    Args:
        ema12 (np.array): The 12-day ema which is used to get the flips
        ema26 (np.array): The 26-day ema which is used to get the flips

    Returns:
        list: The list of flips between the 12-day ema and 26-day ema.
        0 is considered as no flip and 1 is considered as a flip.
    """
    temp = []
    shortmore = None
    for short, mid in zip(ema12, ema26):
        if shortmore is None:
            shortmore = short>mid
        elif shortmore and short<mid:
            temp.append(1)
            shortmore = False
            continue
        elif not shortmore and short>mid:
            temp.append(1)
            shortmore = True
            continue
        temp.append(0)
    return temp


def get_relavant_values(start_date: str, end_date: str, stock_symbol: str,
                        information_keys: List[str]) -> Tuple[Dict, np.array, str, str]:
    """
    The purpose of this function is to get the relevant values between the start and end date range
    as well as the corrected dates.

    Args:
        start_date (str): The minimum start date which is used to get the relevant values
        end_date (str): The maximum end date which is used to get the relevant values
        stock_symbol (str): The stock symbol which is used to get the relevant values
        information_keys (list[str]): The information keys which are used to get the relevant values

    Returns:
        Tuple[dict, np.array, str, str]: The relevant indicators in the
        form of a dict, and list, start date, and end date
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
        raise ValueError(f"Run getInfo.py with start date before {start_date}\n before {end_date}")
    if end_date in other_vals['Dates']:
        i = other_vals['Dates'].index(end_date)
        other_vals['Dates'] = other_vals['Dates'][:i]
        for key in information_keys:
            if key in excluded_values:
                continue
            other_vals[key] = other_vals[key][:i]
    else:
        raise ValueError(f"Run getInfo.py with end date after {start_date}\n and before {end_date}")

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
    return other_vals, filtered, start_date, end_date


def get_scaler(num: float, data: List) -> float:
    """
    Scales the list between 0 and 1 using the `min` and `max` values in the data.
    Used to scale data.

    Args:
        num (float): The number which is to be scaled
        data (list): The data which is used to get the `min` and `max`

    Returns:
        float: The scaler number
    """
    low, high = min(data), max(data)
    return (num - low) / (high - low)


def supertrends(data: pd.DataFrame, factor: int, period: int) -> pd.Series:
    """Returns the `supertrend` indicator"""
    atr = data['High'] - data['Low']
    atr = atr.rolling(window=period).mean()

    upper_band = data['Close'] + (factor * atr)
    lower_band = data['Close'] - (factor * atr)

    trend = np.where(data['Close'] > lower_band, 1, 0)
    trend = np.where(data['Close'] < upper_band, -1, trend)

    signal = pd.Series(trend).diff()
    return signal


def kumo_cloud(data: pd.DataFrame, conversion_period: int=9,
               base_period: int=26, lagging_span2_period: int=52,
               displacement: int=26) -> pd.Series:
    """Gets a pd.Series of where `data['Close']` is above or bellow the kumo cloud"""
    # Calculate conversion line (Tenkan-sen)
    top_conversion = data['High'].rolling(window=conversion_period).max()
    bottom_conversion = data['Low'].rolling(window=conversion_period).min()
    conversion_line = (top_conversion + bottom_conversion) / 2

    # Calculate base line (Kijun-sen)
    top_base = data['High'].rolling(window=base_period).max()
    bottom_base = data['Low'].rolling(window=base_period).min()
    base_line = (top_base + bottom_base) / 2

    # Calculate leading span A (Senkou Span A)
    leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)

    # Calculate leading span B (Senkou Span B)
    span_b_max = data['High'].rolling(window=lagging_span2_period).max()
    span_b_min = data['Low'].rolling(window=lagging_span2_period).min()
    leading_span_b = ((span_b_max + span_b_min) / 2).shift(displacement)

    # Concatenate leading span A and leading span B
    span_concat = pd.concat([leading_span_a, leading_span_b], axis=1)

    # Calculate cloud's top and bottom lines
    cloud_top = span_concat.max(axis=1)
    cloud_bottom = span_concat.min(axis=1)

    cloud_status = np.where(data['Close'] < cloud_bottom, -1, 0)
    cloud_status = np.where(data['Close'] > cloud_top, 1, cloud_status)

    return cloud_status
