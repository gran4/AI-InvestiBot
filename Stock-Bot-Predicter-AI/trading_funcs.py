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

from typing import Optional, List, Tuple, Dict, Iterable
from numbers import Number
from typing import Optional, List, Tuple, Dict, Iterable
from numbers import Number
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
    'check_for_holidays',
    'get_relavant_values',
    'get_scaler',
    'supertrends',
    'kumo_cloud',
    'is_floats'
)


#values that do not go from day to day
#EX: earnings comeout every quarter
excluded_values = (
    "Dates",
    "earnings dates",
    "earning diffs"
)


company_symbols = (
    #"AAPL",
    #"GOOG",
    "TLSA",
    "META",
    "AMZN",
    "DIS",
    "BRK-B",
    "BA",
    "HD",
    "NKE",
    "SBUX",
    "NVDA",
    "CVS",
    "MSFT",
    "NFLX",
    "MCD",
    "KO",
    "V",
    "IBM",
    "WMT",
    "XOM",
    "ADBE",
    "T",
    "GE"
)


def create_sequences(data: np.ndarray, num_days: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    The purpose of this function is to create sequences and labels which are implemented
    into the model during fitting. This is done by iterating through the data and appending
    the data to the sequences and labels list.

    Args:
        data (np.ndarray): The data which is used to create the sequences and labels
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
            - diffs (list) The earning diffserences bettween the expected
            and actual earnings per share
    """
    #_________________deletes earnings before start and after end______________________#
    start = 0
    end = -1 # till the end if nothing
    for date in dates:
        if date < start_date:
            end = dates.index(date)
            break
    for date in dates:
        if date > end_date:
            start = dates.index(date)
            break
    start = 0
    end = -1 # till the end if nothing
    for date in dates:
        if date < start_date:
            end = dates.index(date)
            break
    for date in dates:
        if date > end_date:
            start = dates.index(date)
            break
    dates = dates[start:end]
    diffs = diffs[start:end]

    #_________________Fill Data out with 0s______________________#
    filled_dates = []
    filled_earnings = []

    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

    # Fill out the list to match the start and end date
    while current_date <= end_datetime:
        filled_dates.append(current_date)
        if current_date in dates:
            existing_index = dates.index(current_date)
            filled_earnings.append(diffs[existing_index])
        else:
            filled_earnings.append(0)
        current_date += timedelta(days=1)
    return dates, diffs


def process_flips(series1: Iterable[Number], series2: Iterable[Number]) -> List[int]:
def process_flips(series1: Iterable[Number], series2: Iterable[Number]) -> List[int]:
    """
    The purpose of this function is to return a list of the flips bettween 2 Iterables. It
    is used for the MACD Model and Impulse MACD Model for 12/26 day ema flips and
    MACD/Signal line flips respectivly.
    The purpose of this function is to return a list of the flips bettween 2 Iterables. It
    is used for the MACD Model and Impulse MACD Model for 12/26 day ema flips and
    MACD/Signal line flips respectivly.

    Args:
        series1 (Iterable[Number]): The 1st series which is used to get the flips
        series2 (Iterable[Number]): The 2nd series which is used to get the flips
        series1 (Iterable[Number]): The 1st series which is used to get the flips
        series2 (Iterable[Number]): The 2nd series which is used to get the flips

    Returns:
        list: The list of flips between the 1st and 2nd series
        list: The list of flips between the 1st and 2nd series
        0 is considered as no flip and 1 is considered as a flip.
    """
    temp = []
    shortmore = series1[0] > series2[0]
    shortmore = series1[0] > series2[0]

    for short, mid in zip(series1, series2):
    for short, mid in zip(series1, series2):
        if (shortmore and short<mid) or (not shortmore and short>mid):
            temp.append(1)
            shortmore = not shortmore # flip
        else:
            temp.append(0)
    return temp


def check_for_holidays(start_date: str, end_date: str) -> Tuple[str, str]:
    """Shifts start and end so they are a stock trading day to stop errors"""
    #_________________Check if start or end is holiday______________________#
    nyse = get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    #_________________Change if it is a holiday______________________#
    start_datetime = pd.to_datetime(start_date).date()
    if start_datetime not in schedule.index:
        # Find the next trading day
        next_trading_day = nyse.valid_days(start_date=start_datetime, end_date=end_date)[0]
        start_date = next_trading_day.date().strftime('%Y-%m-%d')

    end_datetime = pd.to_datetime(end_date).date()
    if end_datetime not in schedule.index:
        end_date = schedule.index[-1].date().strftime('%Y-%m-%d')
    
    return start_date, end_date


def get_relavant_values(start_date: str, end_date: str, stock_symbol: str,
                        information_keys: List[str], scaler_data: Optional[Dict]
                        ) -> Tuple[Dict, np.ndarray, Dict, str, str]:
                        ) -> Tuple[Dict, np.ndarray, Dict, str, str]:
    """
    The purpose of this function is to get the relevant values between the start and end date range
    as well as the corrected dates.

    Args:
        start_date (str): The minimum start date which is used to get the relevant values
        end_date (str): The maximum end date which is used to get the relevant values
        stock_symbol (str): The stock symbol which is used to get the relevant values
        information_keys (list[str]): The information keys which are used to get the relevant values

    Returns:
        Tuple[dict, np.ndarray, str, str]: The relevant indicators in the
        form of a dict, and list, start date, and end date
    """
    start_date, end_date = check_for_holidays(start_date, end_date)

    #_________________Load info______________________#
    with open(f'Stocks/{stock_symbol}/info.json', 'r') as file:
    with open(f'Stocks/{stock_symbol}/info.json', 'r') as file:
        other_vals: Dict = json.load(file)

    #_________________Make Data fit between start and end date______________________#
    if start_date in other_vals['Dates']:
        i = other_vals['Dates'].index(start_date)
        other_vals['Dates'] = other_vals['Dates'][i:]
        for key in information_keys:
            if key in excluded_values:
                continue
            other_vals[key] = other_vals[key][i:]
    else:
        raise ValueError(f"Run getInfo.py with start date before {start_date} and {end_date}")
    if end_date in other_vals['Dates']:
        i = other_vals['Dates'].index(end_date)
        other_vals['Dates'] = other_vals['Dates'][:i]
        for key in information_keys:
            if key in excluded_values:
                continue
            other_vals[key] = other_vals[key][:i]
    else:
        raise ValueError(f"Run getInfo.py with end date after {start_date} and {end_date}")

    #_________________Process earnings______________________#
    if "earning diffs" in information_keys:
        dates = other_vals['earnings dates']
        diffs = other_vals['earning diffs']
 
        dates, diffs = process_earnings(dates, diffs, start_date, end_date)
        other_vals['earnings dates'] = dates
        other_vals['earning diffs'] = diffs

    #_________________Scale Data______________________#
    temp = {}
    temp = {}
    for key in information_keys:
        if len(other_vals[key]) == 0:
            continue
        if type(other_vals[key][0]) not in (float, int):
            continue

        if scaler_data is None:
            min_val = min(other_vals[key])
            diff = max(other_vals[key])-min_val
            temp[key] = {'min': min_val, 'diff': diff}
            min_val = min(other_vals[key])
            diff = max(other_vals[key])-min_val
            temp[key] = {'min': min_val, 'diff': diff}
        else:
            min_val = scaler_data[key]['min']
            diff = scaler_data[key]['diff']
        if diff != 0: # Rare, extreme cases
            other_vals[key] = [(x - min_val) / diff for x in other_vals[key]]
    scaler_data = temp # change it if value is `None`
            min_val = scaler_data[key]['min']
            diff = scaler_data[key]['diff']
        if diff != 0: # Rare, extreme cases
            other_vals[key] = [(x - min_val) / diff for x in other_vals[key]]
    scaler_data = temp # change it if value is `None`

    # Convert the dictionary of lists to a NumPy array
    filtered = [other_vals[key] for key in information_keys if key not in excluded_values]
    filtered = np.transpose(filtered) # type: ignore[assignment]

    return other_vals, filtered, scaler_data, start_date, end_date # type: ignore[return-value]
    return other_vals, filtered, scaler_data, start_date, end_date # type: ignore[return-value]


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


def supertrends(data: pd.DataFrame, period: int=10, factor: int=3):
    # Calculate the average true range (ATR)
    tr1 = data["High"] - data["Low"]
    tr2 = abs(data["High"] - data.shift()["Close"])
    tr3 = abs(data["Low"] - data.shift()["Close"])
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    
    # Calculate the basic upper and lower bands
    upper_band = data["Close"].rolling(period).mean() + (factor * atr)
    lower_band = data["Close"].rolling(period).mean() - (factor * atr)

    # Calculate the SuperTrend values using np.select()
    conditions = [
        data["Close"] > upper_band,
        data["Close"] < lower_band
    ]
    choices = [1, -1]
    super_trend = np.select(conditions, choices, default=0)

    return super_trend



def kumo_cloud(data: pd.DataFrame, conversion_period: int=9,
               base_period: int=26, lagging_span2_period: int=52,
               displacement: int=26) -> np.ndarray:
    """Gets a np.ndarray of where `data['Close']` is above or bellow the kumo cloud"""
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


def is_floats(array: List) -> bool:
    """Checks if the list is made of floats"""
    for i in array:
        return type(i) == float
    return False # for cases were the length is 0
