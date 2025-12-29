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
import os
import time

from typing import Optional, List, Tuple, Dict, Iterable
from numbers import Number
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from pandas_market_calendars import get_calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests

__all__ = (
    'non_daily',
    'non_daily_no_use',
    'indicators_to_add_noise_to',
    'company_symbols',
    'download_stock_history',
    'download_stock_history_alphavantage',
    'download_stock_history_stooq',
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
    'calculate_percentage_movement_together',
    'plot'
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
    "MSFT",# 5:59
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


def compute_ema_spread(data: pd.Series, fast: int = 10, slow: int = 40) -> pd.Series:
    fast_ema = data.ewm(span=fast, adjust=False).mean()
    slow_ema = data.ewm(span=slow, adjust=False).mean()
    return fast_ema - slow_ema


def compute_volume_surge(data: pd.Series, window: int = 20, factor: float = 2.0) -> pd.Series:
    rolling_mean = data.rolling(window=window, min_periods=1).mean()
    surge = data / rolling_mean
    return surge.fillna(1.0).clip(0.0, factor)


def compute_atr_series(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean().fillna(0.0)


def download_stock_history(symbol: str,
                           interval: str = "1d",
                           period: str = "max",
                           max_retries: int = 5,
                           base_retry_delay: float = 5.0,
                           alphavantage_api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Download historical data for a ticker with basic retry logic to reduce
    the number of transient Yahoo Finance failures that surface as
    JSONDecodeError/No timezone found errors.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            history = yf.download(
                symbol,
                interval=interval,
                period=period,
                auto_adjust=False,
                threads=False,
                progress=False,
                group_by='column',
            )
            if not history.empty:
                return history
            last_error = RuntimeError("Empty response received from Yahoo Finance.")
        except Exception as exc:  # pragma: no cover - network stack specific
            last_error = exc

        if attempt < max_retries:
            wait_time = base_retry_delay * attempt
            print(f"[download_stock_history] Retrying {symbol} in {wait_time:.1f}s "
                  f"(attempt {attempt}/{max_retries}) due to: {last_error}")
            time.sleep(wait_time)
    print(f"[download_stock_history] Falling back to AlphaVantage for {symbol} after "
          f"{max_retries} failed Yahoo attempts.")
    fallback_key = alphavantage_api_key or os.getenv("ALPHAVANTAGE_API_KEY")
    alpha_data = download_stock_history_alphavantage(symbol, api_key=fallback_key)
    if not alpha_data.empty:
        return alpha_data

    print(f"[download_stock_history] AlphaVantage unavailable for {symbol}. "
          "Using Stooq daily history as a last resort.")
    stooq_data = download_stock_history_stooq(symbol)
    if stooq_data.empty:
        raise ConnectionError(f"Failed to retrieve historical data for {symbol} "
                              f"after Yahoo, AlphaVantage, and Stooq fallbacks.") from last_error
    return stooq_data


def download_stock_history_alphavantage(symbol: str, api_key: Optional[str]) -> pd.DataFrame:
    """
    Fallback data download that uses the TIME_SERIES_DAILY_ADJUSTED endpoint.
    AlphaVantage enforces strict limits (5 calls/minute, 500/day) so this should
    only run when Yahoo Finance repeatedly fails.
    """
    if not api_key:
        print("[download_stock_history] No AlphaVantage API key supplied; skipping fallback.")
        return pd.DataFrame()
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # pragma: no cover - external service
        print(f"[download_stock_history] AlphaVantage request failed for {symbol}: {exc}")
        return pd.DataFrame()

    if "Time Series (Daily)" not in data:
        print(f"[download_stock_history] AlphaVantage returned no data for {symbol}: {data}")
        return pd.DataFrame()

    daily_series = data["Time Series (Daily)"]
    records = []
    for date_str, values in daily_series.items():
        try:
            records.append({
                "Date": pd.to_datetime(date_str),
                "Open": float(values["1. open"]),
                "High": float(values["2. high"]),
                "Low": float(values["3. low"]),
                "Close": float(values["4. close"]),
                "Adj Close": float(values["5. adjusted close"]),
                "Volume": int(float(values["6. volume"])),
            })
        except (KeyError, ValueError) as exc:
            print(f"[download_stock_history] Skipping invalid AlphaVantage row for {symbol} "
                  f"on {date_str}: {exc}")
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records).sort_values("Date")
    df.set_index("Date", inplace=True)
    return df


def download_stock_history_stooq(symbol: str) -> pd.DataFrame:
    """
    Download daily history from Stooq (https://stooq.com/). This endpoint does
    not require an API key but only provides end-of-day data.
    """
    # Stooq symbols are lowercase and suffixed by market, e.g., aapl.us
    stooq_symbol = symbol.lower()
    if "." not in stooq_symbol:
        stooq_symbol = f"{stooq_symbol}.us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    try:
        df = pd.read_csv(url)
    except Exception as exc:  # pragma: no cover - network I/O
        print(f"[download_stock_history] Stooq request failed for {symbol}: {exc}")
        return pd.DataFrame()

    if df.empty or 'Date' not in df:
        print(f"[download_stock_history] Stooq returned no data for {symbol}.")
        return pd.DataFrame()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.rename(columns=str.title, inplace=True)  # ensure columns like 'Open', 'Close'
    return df.sort_index()

def find_best_number_of_years(symbol: str, stock_data: Optional[pd.DataFrame]=None, max_years_back: Optional[int]=None):
    """
    NOTE: NOT PERFECT on leap years,
        Small fix may not be worth the time
    """
    best_years = 3
    if stock_data is None:
        stock_data = download_stock_history(symbol)

    today = date.today().strftime('%Y-%m-%d')
    today_datetime = datetime.strptime(today, '%Y-%m-%d')

    history = stock_data.copy()
    history.index = pd.to_datetime(history.index)
    if getattr(history.index, "tz", None) is not None:
        history.index = history.index.tz_localize(None)

    if history.empty:
        raise RuntimeError(f"No historical data to evaluate ATR for {symbol}.")

    iso_date_dt = history.index[0].to_pydatetime()
    if max_years_back is None:
        max_years_back = today_datetime - iso_date_dt
        max_years_back = max_years_back.days // 365

    best_atr = -float('inf')
    for years in range(4, max_years_back): #ignores 1st year of ipo
        start_date = today_datetime-relativedelta(years=years)

        sliced_history = history[history.index >= start_date]
        if sliced_history.empty:
            continue
        atr = calculate_average_true_range(sliced_history.copy())

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
                        scaler_data: Optional[Dict]=None, scale:bool=False,
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
    leak_prone = {'Future Close'}
    overlap = leak_prone.intersection(information_keys)
    if overlap:
        raise ValueError(f"{overlap.pop()} is a future-looking feature and cannot be used "
                         "as an input. Remove it from information_keys to avoid data leakage.")

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

    max_available_date = other_vals['Dates'][-1]
    if end_date > max_available_date:
        print(f"[get_relavant_values] Clamping end_date from {end_date} to latest available {max_available_date}.")
        end_date = max_available_date

    min_available_date = other_vals['Dates'][0]
    if start_date < min_available_date:
        print(f"[get_relavant_values] Clamping start_date from {start_date} to earliest available {min_available_date}.")
        start_date = min_available_date

    start_date, end_date = check_for_holidays(start_date, end_date)
    if start_date in other_vals['Dates']:
        i = other_vals['Dates'].index(start_date)
        other_vals['Dates'] = other_vals['Dates'][i:]
        for key in information_keys:
            if key in non_daily:
                continue
            if key in other_vals:
                other_vals[key] = other_vals[key][i:]
    else:
        raise ValueError(f"start date is not in data\nRun getInfo.py with start date before {start_date} and {end_date}")

    if end_date in other_vals['Dates']:
        i = other_vals['Dates'].index(end_date)
        other_vals['Dates'] = other_vals['Dates'][:i]
        for key in information_keys:
            if key in non_daily:
                continue
            if key in other_vals:
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

    close_series = pd.Series(other_vals.get('Close', []))
    if not close_series.empty:
        if "returns_zscore" in information_keys and "returns_zscore" not in other_vals:
            returns = close_series.pct_change().fillna(0)
            rolling_mean = returns.rolling(window=20, min_periods=5).mean()
            rolling_std = returns.rolling(window=20, min_periods=5).std().replace(0, np.nan)
            zscore = ((returns - rolling_mean) / rolling_std).fillna(0)
            other_vals["returns_zscore"] = zscore.tolist()

        if "volatility_14" in information_keys and "volatility_14" not in other_vals:
            returns = close_series.pct_change().fillna(0)
            volatility = returns.rolling(window=14, min_periods=5).std().fillna(0)
            other_vals["volatility_14"] = volatility.tolist()

        if "trend_strength" in information_keys and "trend_strength" not in other_vals:
            fast = close_series.ewm(span=50, adjust=False).mean()
            slow = close_series.ewm(span=200, adjust=False).mean()
            baseline = close_series.replace(0, np.nan)
            strength = ((fast - slow) / baseline).replace([np.inf, -np.inf], 0).fillna(0)
            other_vals["trend_strength"] = strength.tolist()

        if "ema_spread_10_40" in information_keys and "ema_spread_10_40" not in other_vals:
            spread = compute_ema_spread(close_series, fast=10, slow=40)
            other_vals["ema_spread_10_40"] = spread.fillna(0).tolist()

        if "atr_14" in information_keys and "atr_14" not in other_vals:
            high = other_vals.get('High', [])
            low = other_vals.get('Low', [])
            close_vals = other_vals.get('Close', [])
            if high and low and close_vals:
                df = pd.DataFrame({'High': high, 'Low': low, 'Close': close_vals})
                atr_series = compute_atr_series(df, window=14)
                other_vals["atr_14"] = atr_series.tolist()
            else:
                other_vals["atr_14"] = [0.0] * len(close_series)

        if "volume_surge" in information_keys and "volume_surge" not in other_vals:
            volume = pd.Series(other_vals.get('Volume', [0] * len(close_series)))
            surge = compute_volume_surge(volume)
            other_vals["volume_surge"] = surge.tolist()

    if "earnings_flag" in information_keys and "earnings_flag" not in other_vals:
        earnings_dates = other_vals.get("earnings dates", [])
        parsed_earnings = [datetime.strptime(date, "%Y-%m-%d") for date in earnings_dates]
        earnings_set = []
        for e_date in parsed_earnings:
            for offset in range(-3, 4):
                earnings_set.append((e_date + relativedelta(days=offset)).strftime("%Y-%m-%d"))
        earnings_lookup = set(earnings_set)
        earnings_flag = [1 if date in earnings_lookup else 0 for date in other_vals['Dates']]
        other_vals["earnings_flag"] = earnings_flag
    stats_reference = scaler_data
    if scale:
        new_stats: Dict[str, Dict[str, float]] = {}
        for key in information_keys:
            if key in non_daily_no_use:
                continue
            values = other_vals.get(key, [])
            if not values or not isinstance(values[0], (float, int)):
                continue

            stats_source = stats_reference.get(key) if stats_reference else None
            if stats_source is None:
                min_val = min(values)
                diff = max(values) - min_val
                new_stats[key] = {'min': min_val, 'diff': diff}
            else:
                min_val = stats_source['min']
                diff = stats_source['diff']

            if diff != 0:
                scale_fn = lambda x, mn=min_val, df=diff: (x - mn) / df
                other_vals[key] = [scale_fn(x) for x in values]
            if key in scale_indicators:
                scaler = scale_indicators[key]
                other_vals[key] = [x * scaler for x in other_vals[key]]

        if stats_reference is None:
            stats_reference = new_stats

    # Convert the dictionary of lists to a NumPy array
    filtered_arrays = []
    lengths: List[int] = []
    for key in information_keys:
        if key in non_daily_no_use:
            continue
        values = other_vals.get(key, [])
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        lengths.append(arr.shape[0])
        filtered_arrays.append(arr)
    if not filtered_arrays:
        raise ValueError("No numeric features available for filtering.")
    min_length = min(lengths)
    if min_length == 0:
        raise ValueError("Numeric features contain empty arrays.")
    filtered = np.stack([arr[:min_length] for arr in filtered_arrays], axis=1)
    return other_vals, filtered, stats_reference # type: ignore[return-value]


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

def plot(data):
    """
    Plots any np.array that you give in
    Purely for testing
    """
    days_train = [i for i in range(data.shape[0])]
    data = data[:, 0]
    # Plot the actual and predicted prices
    plt.figure(figsize=(18, 6))

    predicted_test = plt.plot(days_train, data, label='Predicted Test')
    plt.title(f'TITLE')
    plt.xlabel("X")
    plt.ylabel("Y")

    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(7))

    plt.legend(
        [predicted_test[0]],
        ['Data']
    )
    plt.show()
