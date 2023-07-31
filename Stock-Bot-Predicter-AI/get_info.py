"""
Name:
    getInfo.py

Purpose:
    This module provides utility functions that retrieve information for the
    stock bot. This includes getting the earnings history, liquidity spikes,
    and the stock price.

Author:
    Grant Yul Hur

See also:
    Other modules that use the getInfo module -> Models.py, trading_funcs.py
"""

import requests
import math
import json
import os
import time
from typing import Optional, List, Tuple
from datetime import datetime, date

import yfinance as yf
import numpy as np
import pandas as pd

from trading_funcs import (
    company_symbols,
    find_best_number_of_years,
    process_flips,
    supertrends,
    kumo_cloud
)


__all__ = (
    'get_earnings_history',
    'date_time_since_ref',
    'earnings_since_time',
    'modify_earnings_dates',
    'get_liquidity_spikes',
    'calculate_momentum_oscillator',
    'get_historical_info'
)


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
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={company_ticker}&apikey=0VZ7ORHBEY9XJGXK"
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


def date_time_since_ref(date_object: datetime, reference_date: datetime) -> int:
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
        date_time_since_ref(date, reference_date)
    """
    date_object = datetime.strptime(start_date, "%Y-%m-%d")
    # Convert the datetime object back to a string in the desired format
    converted_date = date_object.strftime("%b %d, %Y")
    reference_date = datetime.strptime(converted_date, "%b %d, %Y")
    return [date_time_since_ref(date, reference_date) for date in dates]


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


def get_liquidity_spikes(data, z_score_threshold: float=2.0,
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


def piecewise_parabolic_weight(years, peak_year, peak_weight, decay_rate):
    if years < peak_year:
        return years ** 2
    else:
        return peak_weight - decay_rate * (years - peak_year) ** 2


def get_historical_info() -> None:
    """
    This function gets the historical info for the given company symbols.
    
    It uses many functions from other modules to process
    historical data and run models on them.
    """
    today = date.today().strftime("%Y-%m-%d")

    period = 14
    for company_ticker in company_symbols:
        print(company_ticker)
        ticker = yf.Ticker(company_ticker)
        #_________________ GET Data______________________#
        # Retrieve historical data for the ticker using the `history()` method
        stock_data = ticker.history(interval="1d", period='max')

        relevant_years = find_best_number_of_years(company_ticker, stock_data=stock_data)
        print(relevant_years)
        num_days = math.log(relevant_years / 60) * 60

        with open(f'Stocks/{company_ticker}/dynamic_tuning.json', 'w') as json_file:
            json.dump({
                'relevant_years': relevant_years,
                'num_days': num_days,
            }, json_file)
        continue

        if len(stock_data) == 0:
            raise ConnectionError("Failed to get stock data. Check your internet")

        #_________________MACD Data______________________#
        ema12 = stock_data['Close'].ewm(span=12).mean()
        ema26 = stock_data['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        histogram = macd-signal_line
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        histogram = macd-signal_line
        ema200 = stock_data['Close'].ewm(span=200).mean()

        #_________________Basically Impulse MACD______________________#
        change = stock_data['Close'].diff()
        change.fillna(change.iloc[1], inplace=True)
        momentum = change.rolling(window=10, min_periods=1).sum()
        momentum.fillna(momentum.iloc[1], inplace=True)

        #_________________Breakout Model______________________#
        gain = change.apply(lambda x: x if x > 0 else 0)  # Positive changes
        loss = change.apply(lambda x: abs(x) if x < 0 else 0)  # Negative changes
        avg_gain = gain.rolling(window=14, min_periods=1).mean()  # 14-day average gain
        avg_loss = loss.rolling(window=14, min_periods=1).mean()  # 14-day average loss

        #RSI Strat
        relative_strength = avg_gain / avg_loss
        relative_strength_index = 100 - (100 / (1 + relative_strength))
        relative_strength_index.fillna(relative_strength_index.iloc[1], inplace=True)

        volatility = stock_data['Close'].diff().abs()  # Calculate price volatility
        # Calculate the initial TRAMA with the specified period
        trama = stock_data['Close'].rolling(window=period, min_periods=1).mean()
        trama = trama + (volatility * 0.1)  # Adjust the TRAMA by adding 10% of the volatility

        bollinger_middle = stock_data['Close'].rolling(window=20, min_periods=1).mean()
        std_dev = stock_data['Close'].rolling(window=20, min_periods=1).std()

        bollinger_upper = bollinger_middle + (2 * std_dev)
        bollinger_lower = bollinger_middle - (2 * std_dev)
        above_bollinger = np.where(stock_data['Close'] > bollinger_upper, 1, 0)
        bellow_bollinger = np.where(stock_data['Close'] < bollinger_lower, 1, 0)

        #_________________Reversal______________________#
        gradual_liquidity_spike = get_liquidity_spikes(stock_data['Volume'], gradual=True)
        liquidity_spike3 = get_liquidity_spikes(stock_data['Volume'], z_score_threshold=4)
        momentum_oscillator = calculate_momentum_oscillator(stock_data['Close'])

        #_________________Process all flips______________________#
        ema_flips = process_flips(ema12.values, ema26.values)
        signal_flips = process_flips(macd, signal_line)
        #_________________Process all flips______________________#
        ema_flips = process_flips(ema12.values, ema26.values)
        signal_flips = process_flips(macd, signal_line)

        #_______________SuperTrendsModel______________#
        super_trend1 = supertrends(stock_data, 3, 12)
        super_trend2 = supertrends(stock_data, 2, 11)
        super_trend3 = supertrends(stock_data, 1, 10)

        kumo_status = kumo_cloud(stock_data)

        #earnings stuffs
        earnings_dates, earnings_diff = get_earnings_history(company_ticker)


        #Do more in the model since we do not know the start or end, yet
        dates = stock_data.index.strftime('%Y-%m-%d').tolist()


        #_________________Process to json______________________#
        converted_data = {
            'Dates': dates,
            'Volume': stock_data['Volume'].values.tolist(),
            'Close': stock_data['Close'].values.tolist(),
            '12-day EMA': ema12.values.tolist(),
            '26-day EMA': ema26.values.tolist(),
            'MACD': signal_line.values.tolist(),
            'Signal Line': signal_line.values.tolist(),
            'Histogram': histogram.values.tolist(),
            '200-day EMA': ema200.values.tolist(),
            'ema_flips': ema_flips,
            'signal_flips':signal_flips,
            'ema_flips': ema_flips,
            'signal_flips':signal_flips,
            'supertrend1': super_trend1.tolist(),
            'supertrend2': super_trend2.tolist(),
            'supertrend3': super_trend3.tolist(),
            'kumo_cloud': kumo_status.tolist(),
            'Momentum': momentum.values.tolist(),
            'Change': change.values.tolist(),
            'RSI': relative_strength_index.values.tolist(),
            'TRAMA': trama.values.tolist(),
            'volatility': volatility.values.tolist(),
            'Bollinger Middle': bollinger_middle.values.tolist(),
            'Above Bollinger': above_bollinger.astype(int).tolist(),
            'Bellow Bollinger': bellow_bollinger.astype(int).tolist(),
            'gradual-liquidity spike': gradual_liquidity_spike.values.tolist(),
            '3-liquidity spike': liquidity_spike3.values.tolist(),
            'momentum_oscillator': momentum_oscillator.values.tolist(),
            'earnings dates': earnings_dates,
            'earning diffs': earnings_diff
        }

        if not os.path.exists(f'Stocks/{company_ticker}'):
            os.makedirs(f'Stocks/{company_ticker}')

        with open(f'Stocks/{company_ticker}/info.json', 'w') as json_file:
            json.dump(converted_data, json_file)

        relevant_years = find_best_number_of_years(company_ticker, stock_data=stock_data)
        print(relevant_years)
        num_days = math.log(relevant_years / 60) * 60
        time.sleep(10231)
        with open(f'Stocks/{company_ticker}/dynamic_tuning.json', 'w') as json_file:
            json.dump({
                'relevant_years': relevant_years,
                'num_days': num_days,
            }, json_file)


if __name__ == '__main__':
    get_historical_info()
