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
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from pandas_market_calendars import get_calendar

import numpy as np
import pandas as pd

__all__ = (
    'non_daily',
    'non_daily_no_use',
    'indicators_to_add_noise_to',
    'company_symbols',
    'create_sequences',
    'find_best_number_of_years',
    'process_earnings',
    'process_flips',
    'check_for_holidays',
    'get_relavant_values',
    'get_scaler',
    'supertrends',
    'kumo_cloud',
    'is_floats',
    'calculate_percentage_movement_together'
)


#values that do not go from day to day
#EX: earnings comeout every quarter
non_daily = (
    "Dates",
    "earnings dates",
    "earning diffs"
)
non_daily_no_use = (
    "Dates",
    "earnings dates",
)

indicators_to_add_noise_to = (
    'Close',
    'Volume',
    'Momentum',
    'Change',
    'Volatility'
)


indicators_to_scale = (
    'Volume',
    'Close',
    '12-day EMA',
    '26-day EMA',
    'MACD',
    'Signal Line',
    'Histogram',
    '200-day EMA',
    'supertrend1',
    'supertrend2',
    'supertrend3',
    #'kumo_cloud',
    'Momentum',
    'Change',
    'TRAMA',
    #'Volatility',
    'Bollinger Middle',
    #'gradual-liquidity spike',
    'momentum_oscillator',
    #'earning diffs'
)



company_symbols = (
    "AAPL",
    "GOOG",
    # "TLSA",
    # "META",
    # "AMZN",
    # "DIS",
    # "BRK-B",
    # "BA",
    # "HD",
    # "NKE",
    # "SBUX",
    # "NVDA",
    # "CVS",
    # "MSFT",# 5:59
    # "NFLX",
    # "MCD",
    # "KO",
    # "V",
    # "IBM",
    # "WMT",
    # "XOM",
    # "ADBE",
    # "T",
    "GE"
)

scale_indicators = {
    'Close': 2,
    'MACD': 1
}


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


def piecewise_parabolic_weight(years, peak_year):
    #years ** 1.5 is to give a curve
    if years < peak_year:
        return years ** 1.5+years/5
    return peak_year ** 1.5+peak_year/6+(peak_year-years)/10


def calculate_average_true_range(stock_data):
    stock_data['High_Low'] = stock_data['High'] - stock_data['Low']
    stock_data['High_PreviousClose'] = abs(stock_data['High'] - stock_data['Close'].shift())
    stock_data['Low_PreviousClose'] = abs(stock_data['Low'] - stock_data['Close'].shift())
    stock_data['TrueRange'] = stock_data[['High_Low', 'High_PreviousClose', 'Low_PreviousClose']].max(axis=1)
    average_true_range = stock_data['TrueRange'].mean()
    return average_true_range

def find_best_number_of_years(symbol: str, stock_data: Optional[pd.DataFrame]=None, max_years_back: Optional[int]=None):
    """
    NOTE: NOT PERFECT on leap years,
        Small fix may not be worth the time
    """
    best_years = 3
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    if stock_data is None:
        stock_data = ticker.history(interval="1d", period='max')

    today = date.today().strftime('%Y-%m-%d')
    today_datetime = datetime.strptime(today, '%Y-%m-%d')
    
    iso_date = stock_data.index[0].strftime('%Y-%m-%d')
    iso_date = datetime.strptime(iso_date, '%Y-%m-%d')
    if max_years_back is None:
        max_years_back = today_datetime - iso_date
        max_years_back = max_years_back.days // 365

    best_atr = -float('inf')
    for years in range(4, max_years_back): #ignores 1st year of ipo
        start_date = today_datetime-relativedelta(years=years)

        stock_data = ticker.history(interval="1d", start=start_date, end=today)
        atr = calculate_average_true_range(stock_data)#stock_data[iso_date >= start_date])

        atr += piecewise_parabolic_weight(years, max_years_back/4)/10 + piecewise_parabolic_weight(years, max_years_back/6)/30

        if atr > best_atr:
            best_atr = atr
            best_years = years
        #print("NORM: ", atr)
        #print("Best: ", best_atr)
        #print()

    return best_years


def process_earnings(dates: List, diffs: List, start_date: str,
                     end_date: str, iterations: int) -> Tuple[List[str], List[float]]:
    """
    The purpose of this function is to process the earnings between the start and
    end date range, and fill in the 0s for dates without an earnings report. 

    Args:
        dates (list): The dates which are used to get the earnings
        diffs (list): The earnings which are used to get the earnings
        start_date (str): The start date which is used to get the earnings
        end_date (str): The end date which is used to get the earnings
        iterations (int): Time since start bc relative time is inaccurate
    
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
    if start > end:
        return [], []

    dates = dates[start:end]
    diffs = diffs[start:end]

    #_________________Fill Data out with 0s______________________#
    filled_dates = []
    filled_earnings = []

    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    # Fill out the list to match the start and end date
    for i in range(iterations):
        filled_dates.append(current_date)
        if current_date in dates:
            existing_index = dates.index(current_date)
            filled_earnings.append(diffs[existing_index])
        else:
            filled_earnings.append(0)
        current_date += relativedelta(days=1)
    return dates, filled_earnings


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


def get_relavant_values(stock_symbol: str, information_keys: List[str],
                        scaler_data: Optional[Dict]=None, start_date: Optional[str]=None,
                        end_date: Optional[str]=None,
                        ) -> Tuple[Dict, np.ndarray, List]:
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
        form of a dict, np.ndarray, and a list
    """
    #_________________Load info______________________#
    with open(f'Stocks/{stock_symbol}/info.json', 'r') as file:
        other_vals: Dict = json.load(file)

    #fit bettween start and end date
    if start_date is None:
        start_date = other_vals['Dates'][0]
    elif type(start_date) is int:
        start_date = other_vals['Dates'][start_date]

    if end_date is None:
        end_date = other_vals['Dates'][-1]
    elif type(end_date) is int:
        end_date = other_vals['Dates'][end_date]

    start_date, end_date = check_for_holidays(start_date, end_date)
    if start_date in other_vals['Dates']:
        i = other_vals['Dates'].index(start_date)
        other_vals['Dates'] = other_vals['Dates'][i:]
        for key in information_keys:
            if key in non_daily:
                continue
            other_vals[key] = other_vals[key][i:]
    else:
        raise ValueError(f"start date is not in data\nRun getInfo.py with start date before {start_date} and {end_date}")

    if end_date in other_vals['Dates']:
        i = other_vals['Dates'].index(end_date)
        other_vals['Dates'] = other_vals['Dates'][:i]
        for key in information_keys:
            if key in non_daily:
                continue
            other_vals[key] = other_vals[key][:i]
    else:
        raise ValueError(f"end date is not in data\nRun getInfo.py with end date after {start_date} and {end_date}")
    #_________________Process earnings______________________#
    if "earning diffs" in information_keys:
        dates = other_vals['earnings dates']
        diffs = other_vals['earning diffs']

        dates, diffs = process_earnings(dates, diffs, start_date, end_date, len(other_vals['Close']))
        other_vals['earnings dates'] = dates
        other_vals['earning diffs'] = diffs

    #_________________Scale Data______________________#
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
        else:
            min_val = scaler_data[key]['min']
            diff = scaler_data[key]['diff']
        if diff != 0: # Ignore rare, extreme cases
            other_vals[key] = [(x - min_val) / diff for x in other_vals[key]]
        if key in scale_indicators:
            scaler = scale_indicators[key]
            other_vals[key] = [x*scaler for x in other_vals[key]]
    scaler_data = temp # change it if value is `None`

    # Convert the dictionary of lists to a NumPy array
    filtered = [other_vals[key] for key in information_keys if key not in non_daily_no_use]
    filtered = np.transpose(filtered) # type: ignore[assignment]
    return other_vals, filtered, scaler_data# type: ignore[return-value]


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
    atr = calculate_average_true_range(data)
    
    rolling_mean = data["Close"].rolling(period).mean()
    rolling_mean = rolling_mean.fillna(rolling_mean.iloc[period - 1])
    # Calculate the basic upper and lower bands
    upper_band = rolling_mean + (factor * atr)
    lower_band = rolling_mean - (factor * atr)

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
    top_conversion = data['High'].rolling(window=conversion_period, min_periods=1).max()
    bottom_conversion = data['Low'].rolling(window=conversion_period, min_periods=1).min()
    conversion_line = (top_conversion + bottom_conversion) / 2

    # Calculate base line (Kijun-sen)
    top_base = data['High'].rolling(window=base_period, min_periods=1).max()
    bottom_base = data['Low'].rolling(window=base_period, min_periods=1).min()
    base_line = (top_base + bottom_base) / 2

    # Calculate leading span A (Senkou Span A)
    leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)

    # Calculate leading span B (Senkou Span B)
    span_b_max = data['High'].rolling(window=lagging_span2_period, min_periods=1).max()
    span_b_min = data['Low'].rolling(window=lagging_span2_period, min_periods=1).min()
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


def calculate_percentage_movement_together(list1: Iterable, list2: Iterable) -> Tuple[float, float]:
    total = len(list1)
    count_same_direction = 0
    count_same_space = 0

    for i in range(1, total):
        if (list1[i] > list1[i - 1] and list2[i] > list2[i - 1]) or (list1[i] < list1[i - 1] and list2[i] < list2[i - 1]):
            count_same_direction += 1
        if (list1[i] > list1[i - 1] and list2[i] > list1[i - 1]) or (list1[i] < list1[i - 1] and list2[i] < list1[i - 1]):
            count_same_space += 1

    percentage = (count_same_direction / (total - 1)) * 100
    percentage2 = (count_same_space / (total - 1)) * 100
    return percentage, percentage2
