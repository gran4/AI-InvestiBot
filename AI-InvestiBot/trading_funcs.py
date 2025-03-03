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
    Similarly related modules involving use of the trading functions -> E.g Models.py, get_info.py
"""
import requests
import json
import time

from typing import Optional, List, Tuple, Dict, Iterable, Union
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
    'all_indicators',
    'suggested_companies',
    'suggested_strategies',
    'ImpulseMACD_indicators',
    'Reversal_indicators',
    'Earnings_indicators',
    'RSI_indicators',
    'break_out_indicators',
    'super_trends_indicators',
    'create_sequences',
    'get_earnings_history',
    'time_since_ref',
    'earnings_since_time',
    'modify_earnings_dates',
    'get_liquidity_spikes',
    'calculate_momentum_oscillator',
    'find_best_number_of_years',
    'process_earnings',
    'process_flips',
    'check_for_holidays',
    'get_relavant_values',
    'get_scaler',
    'supertrends',
    'supertrendsV2',
    'kumo_cloud',
    'kumo_cloudV2',
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

ImpulseMACD_indicators = ['Close', 'Histogram', 'Momentum', 'Change', 'ema_flips', 'signal_flips', '200-day EMA']
Reversal_indicators = ['Close', 'gradual-liquidity spike', '3-liquidity spike', 'momentum_oscillator']
Earnings_indicators = ['Close', 'earnings dates', 'earning diffs', 'Momentum']
RSI_indicators = ['Close', 'RSI', 'TRAMA']
break_out_indicators = ['Close', 'Bollinger Middle',
    'Above Bollinger', 'Bellow Bollinger', 'Momentum']
super_trends_indicators = ['Close', 'supertrend1', 'supertrend2',
    'supertrend3', '200-day EMA', 'kumo_cloud']
all_indicators = [
    'Close', 'Histogram', 'Momentum', 'Change', 'ema_flips', 'signal_flips',
    '200-day EMA', 'gradual-liquidity spike', '3-liquidity spike',
    'momentum_oscillator', 'RSI', 'TRAMA',
    'Bollinger Middle', 'Above Bollinger', 'Bellow Bollinger', 'supertrend1', 'supertrend2',
    'supertrend3', 'kumo_cloud'
    ]

suggested_strategies = {"ImpulseMACD_indicators": ImpulseMACD_indicators, "Reversal_indicators": Reversal_indicators, "break_out_indicators": break_out_indicators, "super_trends_indicators": super_trends_indicators, 'RSI_indicators': RSI_indicators}



("COST", "TGT", "KMB", "CHD",
    "MNST", "MDLZ", "STZ", "HSY", "EL", "BF-B", "CPB", "HRL", "CLX", "TSN",
    
    # Energy
    "CVX", "SLB", "HAL", "PSX", "MRO", "OXY", "PXD", "VLO", "EOG", "FANG",
    "APA", "COP", "HES", "WMB", "OKE", "KMI", "ENB", "XEC", "LNG", "ET",
    
    # Financial
    "JPM", "BAC", "C", "WFC", "GS", "MS", "PNC", "USB", "TFC", "SCHW",
    "BK", "BLK", "AXP", "COF", "DFS", "VICI", "NTRS", "ICE", "CME", "FIS",
    
    # Healthcare
    "JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "UNH", "HUM", "CI",
    "GILD", "REGN", "VRTX", "SYK", "ABT", "BDX", "EW", "ISRG", "BIIB", "DHR",
    
    # Industrials
    "CAT", "DE", "MMM", "HON", "LMT", "RTX", "GD", "NOC", "CMI", "ETN",
    "EMR", "ITW", "FAST", "URI", "JCI", "DOV", "IEX", "SNA", "ROL", "TT",
    
    # Retail and Other
    "BBY", "DG", "DLTR", "ULTA", "ROST", "TJX", "W", "KR", "FIVE", "CVNA"
)
suggested_companies = (
    "AAPL",
    "GOOG",
    "TLSA",
    "META",
    "AMZN",
    "DIS",
    #"BRK-B",
    "BA",
    "HD",
    "NKE",
    #"SBUX",
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
    "GE",
    "RIVN",
    "PLTR",

    # Technology
    "INTC", "AMD", "CSCO", "ORCL", "HPQ", "TXN", "QCOM", "CRM", "SHOP", "SNOW",
    "PYPL", "SQ", "ZS", "PANW", "NET", "UBER", "LYFT", "DOCU", "CRWD", "WDAY",
    
    # Consumer Goods
    "PG", "PEP", "CL", "KHC", "GIS", "KDP", 
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
        temp = data[i-num_days:i]
        low = np.min(temp, axis=0)
        high = np.max(temp, axis=0)
        diff = high-low
        temp = (temp - low) / (diff)
        temp = np.nan_to_num(temp, nan=0.0)

        sequences.append(temp)
        labels.append(data[i, 0])
    return np.array(sequences), np.array(labels)


def piecewise_parabolic_weight(years, peak_year):
    #years ** 1.5 is to give a curve
    if years < peak_year:
        return years ** 1.5+years/5
    return peak_year ** 1.5+peak_year/6+(peak_year-years)/10



def get_earnings_history(company_ticker: str) -> Tuple[List[str], List[float]]:
    """
    Gets earning history of a company as a list.

    Args:
        company_ticker str: company to get info of
        context Optional[ssl certificate]: ssl certificate to use

    Warning:
        YOU need to process this data later in runtime

    Returns:
        Tuple: of 2 lists made of: Date and EPS_difference, respectively
    """
    # API key is mine, OK since it is only for data retrieval
    # and, I do not use it - gran4
    api_key = "0VZ7ORHBEY9XJGXK"
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={company_ticker}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if len(data.keys()) == 1:
        print('Wait for alpha api to reset(It takes 1min).')
        time.sleep(60)
        response = requests.get(url)
        data = response.json()
    if len(data.keys()) == 1:
        raise RuntimeError("You exceded alpha's api limit of 500 calls per day")

    earnings_dates = []
    earnings_diff = []
    for quarter in data["quarterlyEarnings"]:
        date = quarter["fiscalDateEnding"]

        actual_eps = 0.0
        if quarter["reportedEPS"] != 'None':
            actual_eps = float(quarter["reportedEPS"])

        estimated_eps = 0.0
        if quarter["estimatedEPS"] != 'None':
            estimated_eps = float(quarter["estimatedEPS"])
        earnings_dates.append(date)
        earnings_diff.append(actual_eps-estimated_eps)

    return earnings_dates, earnings_diff


def time_since_ref(date_object: Union[datetime, relativedelta], reference_date: Union[datetime, relativedelta]) -> int:
    """
    Returns the number of days between the given date and the reference date.

    Args:
        date_object (datetime): Date to calculate the difference from.
        reference_date (datetime): Reference date to calculate the difference to.

    Returns:
        int: Number of days between the two dates.
    """
    # Calculate the number of days between the date and the reference date
    return (date_object - reference_date).days

def earnings_since_time(dates: List, start_date: str) -> List[int]:
    """
    This function will return a list of earnings since the list of dates
    and the reference date.

    Args:
        dates (list): list of dates to calculate the difference from.
        start_date (str): Reference date to calculate the difference to.
    
    Returns:
        list: list of earnings since the reference date using
        time_since_ref(date, reference_date)
    """
    date_object = datetime.strptime(start_date, "%Y-%m-%d")
    # Convert the datetime object back to a string in the desired format
    converted_date = date_object.strftime("%b %d, %Y")
    reference_date = datetime.strptime(converted_date, "%b %d, %Y")
    return [time_since_ref(date, reference_date) for date in dates]


def modify_earnings_dates(dates: List, start_date: str) -> List[int]:
    """
    This function will modify the earning dates using the
    earnings_since_time function.

    Args:
        dates (list): list of dates to calculate the difference from.
        start_date (str): Reference date to calculate the difference to.
    
    Returns:
        list: list of earnings since the reference date using
        earnings_since_time(dates, start_date)
    """
    temp = [datetime.strptime(date_string, "%b %d, %Y") for date_string in dates]
    return earnings_since_time(temp, start_date)


def get_liquidity_spikes(data: pd.DataFrame, z_score_threshold: float=2.0,
                         gradual: bool=False) -> pd.Series:
    """
    This function will get the spikes in liquidity for given stock data.

    Args:
        data (pd.Series): Stock data to calculate the spikes from.
        z_score_threshold (float): Threshold to determine if a spike is abnormal.
        gradual (bool): Whether to gradually increase the z-score or not.
    
    Returns:
        pd.Series: Series of abnormal spikes in liquidity returned
        as a scale between 0 and 1.
    """
    # Convert the data to a pandas Series if it's not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    # Calculate the rolling average and standard deviation of bid volume
    window_size = 20
    rolling_average = data.rolling(window_size, min_periods=1).mean()
    rolling_std = data.rolling(window_size, min_periods=1).std()

    # Replace NaN values in rolling_average and rolling_std with zeros
    rolling_average = rolling_average.fillna(0)
    rolling_std = rolling_std.fillna(0)

    # Calculate Z-scores to identify abnormal bid volume spikes
    z_scores = (data - rolling_average) / rolling_std

    if gradual:
        abnormal_spikes = z_scores
        abnormal_spikes.fillna(abnormal_spikes.iloc[1], inplace=True)
    else:
        # Detect abnormal bid volume spikes
        abnormal_spikes = pd.Series(np.where(z_scores > z_score_threshold, 1, 0), index=data.index)
        abnormal_spikes.astype(int)

    return abnormal_spikes


def calculate_momentum_oscillator(data: pd.Series, period: int=14) -> pd.Series:
    """
    Calculates the momentum oscillator for the given data series.

    Args:
        data (pd.Series): Input data series.
        period (int): Number of periods for the oscillator calculation.

    Returns:
        pd.Series: Momentum oscillator values.
    """
    momentum = data.diff(period)  # Calculate the difference between current and n periods ago
    percent_momentum = (momentum / data.shift(period)) * 100  # Calculate momentum as a percentage

    # type: ignore[no-any-return]
    return percent_momentum.fillna(method='bfill')



def calculate_dynamic_atr(stock_data, period=14):
    high_low = stock_data['High'] - stock_data['Low']
    high_close = abs(stock_data['High'] - stock_data['Close'].shift())
    low_close = abs(stock_data['Low'] - stock_data['Close'].shift())
    true_ranges = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close}).max(axis=1)
    atr = true_ranges.rolling(window=period, min_periods=1).mean()
    return atr

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
        atr = calculate_true_range(stock_data['High'], stock_data['Low'], stock_data['Close'])
        atr = sum(atr)/len(atr)
        atr += piecewise_parabolic_weight(years, max_years_back/4)/10 + piecewise_parabolic_weight(years, max_years_back/6)/30

        if atr > best_atr:
            best_atr = atr
            best_years = years
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

def scale_func(information_keys: List[str], data: Dict, scaler_data: Dict) -> Dict:
    temp = {}
    for key in information_keys:
        if len(data[key]) == 0:
            continue
        if type(data[key][0]) not in (float, int):
            continue

        if scaler_data == {}:
            min_val = min(data[key])
            diff = max(data[key])-min_val
            temp[key] = {'min': min_val, 'diff': diff}
        else:
            min_val = scaler_data[key]['min']
            diff = scaler_data[key]['diff']
        if diff != 0: # Ignore rare, extreme cases
            data[key] = [(x - min_val) / diff for x in data[key]]
        if key in scale_indicators:
            scaler = scale_indicators[key]
            data[key] = [x*scaler for x in data[key]]
    return temp

def get_relavant_values(stock_symbol: str, information_keys: List[str],
                        scale: bool = False, scaler_data: Optional[Dict]={},
                        start_date: Optional[str]=None, end_date: Optional[str]=None,
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
    print(start_date, end_date)
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
    if scale:
        scaler_data = scale_func(information_keys, other_vals, scaler_data)
    # Convert the dictionary of lists to a NumPy array
    filtered = [other_vals[key] for key in information_keys if key not in non_daily_no_use]
    filtered = np.transpose(filtered) # type: ignore[assignment]2
    return other_vals, filtered, scaler_data


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


def supertrends(data, period=10, multiplier=3):
    atr = calculate_dynamic_atr(data, period)
    hl2 = (data['High'] + data['Low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    super_trend = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if data['Close'][i] > upper_band[i-1]:
            super_trend[i] = upper_band[i]
        elif data['Close'][i] < lower_band[i-1]:
            super_trend[i] = lower_band[i]
        else:
            super_trend[i] = super_trend[i-1]
    
    return super_trend



def calculate_true_range(high, low, close):
    tr = [0] * len(close)
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return tr

def calculate_wilders_smoothing(values, period):
    smoothed_values = np.zeros(len(values))
    initial_avg = sum(values[:period]) / period
    smoothed_values[period-1] = initial_avg
    for i in range(period, len(values)):
        smoothed_values[i] = (smoothed_values[i-1] * (period - 1) + values[i]) / period
    return smoothed_values

def supertrendsV2(high, low, close, period=7, multiplier=3.0):
    true_range = calculate_true_range(high, low, close)
    atr = calculate_wilders_smoothing(true_range, period)
    
    hl2 = (np.array(high) + np.array(low)) / 2
    upper_band_basic = hl2 + (multiplier * atr)
    lower_band_basic = hl2 - (multiplier * atr)
    
    upper_band = np.zeros(len(close))
    lower_band = np.zeros(len(close))
    supertrend = np.zeros(len(close))
    
    for i in range(1, len(close)):
        upper_band[i] = upper_band_basic[i] if (i == period or close[i-1] <= upper_band[i-1] or upper_band_basic[i] < upper_band[i-1]) else upper_band[i-1]
        lower_band[i] = lower_band_basic[i] if (i == period or close[i-1] >= lower_band[i-1] or lower_band_basic[i] > lower_band[i-1]) else lower_band[i-1]
        
        if close[i] > upper_band[i-1]:
            supertrend[i] = upper_band[i]
        elif close[i] < lower_band[i-1]:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = supertrend[i-1]

    # Fix initial values
    supertrend[:period] = np.nan  # Not enough data to calculate
    return supertrend

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

def kumo_cloudV2(high: list, low: list, close: list, conversion_period: int=9,
               base_period: int=26, lagging_span2_period: int=52,
               displacement: int=26) -> np.ndarray:
    """Gets a np.ndarray of where `close` is above or below the kumo cloud"""
    # Calculate conversion line (Tenkan-sen)
    cloud_status = np.zeros(len(close), dtype=int)  # Initialize the result array
    leading_span_a = [None] * len(close)  # Initialize leading_span_a with None values
    leading_span_b = [None] * len(close)  # Initialize leading_span_b with None values


    for i in range(len(close)):
        # Calculate conversion line (Tenkan-sen)
        if i >= conversion_period:
            top_conversion = max(high[i-conversion_period+1:i+1])
            bottom_conversion = min(low[i-conversion_period+1:i+1])
            conversion_line = (top_conversion + bottom_conversion) / 2

            # Calculate base line (Kijun-sen)
            if i >= base_period:
                top_base = max(high[i-base_period+1:i+1])
                bottom_base = min(low[i-base_period+1:i+1])
                base_line = (top_base + bottom_base) / 2

                # Calculate leading span A (Senkou Span A)
                leading_span_a = [((conversion_line + base_line) / 2)] * displacement + leading_span_a[:i-displacement]

                # Calculate leading span B (Senkou Span B)
                if i >= lagging_span2_period:
                    span_b_max = max(high[i-lagging_span2_period+1:i+1])
                    span_b_min = min(low[i-lagging_span2_period+1:i+1])
                    leading_span_b = ((span_b_max + span_b_min) / 2)
                    leading_span_b = [None] * displacement + leading_span_b[:i-displacement]

                    # Calculate cloud's top and bottom lines
                    cloud_top = max(leading_span_a, leading_span_b)
                    cloud_bottom = min(leading_span_a, leading_span_b)

                    # Determine cloud status for this time step
                    if close[i] < cloud_bottom:
                        cloud_status[i] = -1
                    elif close[i] > cloud_top:
                        cloud_status[i] = 1

    return cloud_status


def is_floats(array: List) -> bool:
    """Checks if the list is made of floats"""
    for i in array:
        return type(i) == float
    return False # for cases were the length is 0


def calculate_percentage_movement_together(list1: Iterable, list2: Iterable) -> Tuple[float, float, float]:
    total = len(list1)
    count_same_direction = 0
    count_exaduration = 0
    count_same_space = 0

    for i in range(1, total):
        if (list1[i] > list1[i - 1] and list2[i] > list2[i - 1]) or (list1[i] < list1[i - 1] and list2[i] < list2[i - 1]):
            count_same_direction += 1
        if (list1[i] > list1[i - 1] and list2[i] > list1[i - 1]) or (list1[i] < list1[i - 1] and list2[i] < list1[i - 1]):
            count_exaduration += 1
        if (list1[i] >= 0 and list2[i] >= 0) or (list1[i] < 0 and list2[i] < 0):
            count_same_space += 1

    percentage = (count_same_direction / (total - 1)) * 100
    percentage2 = (count_exaduration / (total - 1)) * 100
    percentage3 = (count_same_space / (total - 1)) * 100
    return percentage, percentage2, percentage3


def get_indicators_for_date(stock_symbol: str, end_date: str,
                                 information_keys: List[str], cached_info: pd.DataFrame, num_days: int,
                                 scale=False, scaler_data: Dict[str, int] = {}) -> Dict[str, Union[float, str]]:
    """
    This method will return the indicators for the past `num_days` days specified in the
    information keys. It will use the cached information to calculate the indicators
    until the `end_date`.

    Args:
        information_keys (List[str]): tells model the indicators to use
        scaler_data (Dict[str, int]): used to scale indicators
        cached_info (pd.DataFrame): The cached information
        num_days (int): The number of days to calculate the indicators for
        
    Returns:
        dict: A dictionary containing the indicators for the stock data
            Values will be floats except some expections tht need to be
            processed during run time
    """
    num_days *= 2
    stock_data = {}

    stock_data['Close'] = cached_info['Close'].iloc[-num_days:]
    stock_data['High'] = cached_info['High'].iloc[-num_days:]
    stock_data['Low'] = cached_info['Low'].iloc[-num_days:]

    ema12 = cached_info['Close'].ewm(span=12, adjust=False).mean()
    ema26 = cached_info['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    span = 9
    signal_line = macd.rolling(window=span, min_periods=1).mean().iloc[-num_days:]

    change = cached_info['Close'].diff()
    if '12-day EMA' in information_keys:
        stock_data['12-day EMA'] = ema12.iloc[-num_days:]
    if '26-day EMA' in information_keys:
        stock_data['26-day EMA'] = ema26.iloc[-num_days:]
    if 'MACD' in information_keys:
        stock_data['MACD'] = macd.iloc[-num_days:]
    if 'Signal Line' in information_keys:
        stock_data['Signal Line'] = signal_line
    if 'Histogram' in information_keys:
        histogram = macd - signal_line
        stock_data['Histogram'] = histogram.iloc[-num_days:]
    if '200-day EMA' in information_keys:
        ewm200 = cached_info['Close'].ewm(span=200, adjust=False)
        ema200 = ewm200.mean().iloc[-num_days:]
        stock_data['200-day EMA'] = ema200
    change = cached_info['Close'].diff().iloc[-num_days:]
    if 'Change' in information_keys:
        stock_data['Change'] = change.iloc[-num_days:]
    if 'Momentum' in information_keys:
        momentum = change.rolling(window=10, min_periods=1).sum().iloc[-num_days:]
        stock_data['Momentum'] = momentum
    if 'RSI' in information_keys:
        gain = change.apply(lambda x: x if x > 0 else 0)
        loss = change.apply(lambda x: abs(x) if x < 0 else 0)
        avg_gain = gain.rolling(window=14).mean().iloc[-num_days:]
        avg_loss = loss.rolling(window=14).mean().iloc[-num_days:]
        relative_strength = avg_gain / avg_loss
        stock_data['RSI'] = 100 - (100 / (1 + relative_strength))
    if 'TRAMA' in information_keys:
        # TRAMA
        volatility = cached_info['Close'].diff().abs().iloc[-num_days:]
        trama = cached_info['Close'].rolling(window=14).mean().iloc[-num_days:]
        stock_data['TRAMA'] = trama + (volatility * 0.1)
    if 'gradual-liquidity spike' in information_keys:
        # Reversal
        stock_data['gradual-liquidity spike'] = get_liquidity_spikes(
            cached_info['Volume'], gradual=True
        ).iloc[-num_days:]
    if '3-liquidity spike' in information_keys:
        stock_data['3-liquidity spike'] = get_liquidity_spikes(
            cached_info['Volume'], z_score_threshold=4
        ).iloc[-num_days:]
    if 'momentum_oscillator' in information_keys:
        stock_data['momentum_oscillator'] = calculate_momentum_oscillator(
            cached_info['Close']
        ).iloc[-num_days:]
    if 'ema_flips' in information_keys:
        #_________________12 and 26 day Ema flips______________________#
        stock_data['ema_flips'] = process_flips(ema12[-num_days:], ema26[-num_days:])
        stock_data['ema_flips'] = pd.Series(stock_data['ema_flips'])
    if 'signal_flips' in information_keys:
        stock_data['signal_flips'] = process_flips(macd[-num_days:], signal_line[-num_days:])
        stock_data['signal_flips'] = pd.Series(stock_data['signal_flips'])
    if 'earning diffs' in information_keys:
        #earnings stuffs
        earnings_dates, earnings_diff = get_earnings_history(stock_symbol)
            
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        date = end_datetime - relativedelta(days=num_days)

        stock_data['earnings dates'] = []
        stock_data['earning diffs'] = [] # type: ignore[attr]
        low = scaler_data['earning diffs']['min'] # type: ignore[index]
        diff = scaler_data['earning diffs']['diff'] # type: ignore[index]

        for i in range(num_days):
            if not end_date in earnings_dates:
                stock_data['earning diffs'].append(0)
                continue
            i = earnings_dates.index(date)
            scaled = (earnings_diff[i]-low) / diff
            stock_data['earning diffs'].append(scaled)
    bollinger_middle = stock_data['Close'].rolling(window=20, min_periods=1).mean()
    std_dev = stock_data['Close'].rolling(window=20, min_periods=1).std()
    if 'Bollinger Middle' in information_keys:
        stock_data['Bollinger Middle'] = bollinger_middle
    if 'Bellow Bollinger' in information_keys:
        bollinger_lower = bollinger_middle - (2 * std_dev)
        stock_data['Bellow Bollinger'] = np.where(stock_data['Close'] < bollinger_lower, 1, 0)
    if 'Above Bollinger' in information_keys:
        bollinger_upper = bollinger_middle + (2 * std_dev)
        stock_data['Above Bollinger'] = np.where(stock_data['Close'] > bollinger_upper, 1, 0)
    if 'supertrend1' in information_keys:
        stock_data['supertrend1'] = supertrendsV2(stock_data['High'], stock_data['Low'], stock_data['Close'], multiplier=3, period=12)
    if 'supertrend2' in information_keys:
        stock_data['supertrend2'] = supertrendsV2(stock_data['High'], stock_data['Low'], stock_data['Close'], multiplier=2, period=11)
    if 'supertrend3' in information_keys:
        stock_data['supertrend3'] = supertrendsV2(stock_data['High'], stock_data['Low'], stock_data['Close'], multiplier=1, period=10)
    if 'kumo_cloud' in information_keys:
        stock_data['kumo_cloud'] = kumo_cloudV2(stock_data['High'], stock_data['Low'], stock_data['Close'])
    if scale and scaler_data == {}:
        # Scale each column manually
        for column in information_keys:
            if column in non_daily:
                continue
            low = scaler_data[column]['min'] # type: ignore[index]
            diff = scaler_data[column]['diff'] # type: ignore[index]
            column_values = stock_data[column]
            scaled_values = (column_values - low) / diff
            scaled_values = (column_values - low) / diff
            stock_data[column] = scaled_values
    return stock_data



