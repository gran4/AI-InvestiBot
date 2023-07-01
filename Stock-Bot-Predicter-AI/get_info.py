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

import urllib.request
import ssl
import json
from typing import Optional, List, Tuple
from datetime import datetime

from bs4 import BeautifulSoup
import yfinance as yf
import numpy as np
import pandas as pd

from trading_funcs import (
    excluded_values,
    company_symbols,
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


def get_earnings_history(company_ticker: str, context: Optional[ssl.SSLContext] = None
                         ) -> Tuple[List[str], List[float]]:
    """
    Gets earning history of a company as a list.

    Args:
        company_ticker str: company to get info of
        context Optional[ssl certificate]: ssl certificate to use

    Warning:
        IF YOU GET ERROR:
            urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate
            verify failed: unable to get local issuer certificate (_ssl.c:1002)>

        Go to Python3.6 folder (or whatever version of python you're using) > double click
        on "Install Certificates.command" file. :D

        NOTE: ON macOS go to Macintosh HD > Applications > Python3.6
        (or whatever version of python you're using), then follow above

    Warning:
        YOU are probably looking to use get_corrected_earnings_history not this

    Returns:
        Tuple: of 2 lists made of: Date and EPS_difference, respectively
    """
    url = f"https://finance.yahoo.com/quote/{company_ticker}/history?p={company_ticker}"

    # Send a GET request to the URL with certificate verification
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req, context=context)
    html_content = response.read().decode('utf-8')

    # Create a Beautiful Soup object for parsing
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the table containing earnings history
    table = soup.find('table', {'data-test': 'historical-prices'})
    rows = table.find_all('tr') # type: ignore[union-attr]

    earnings_dates, earnings_diff = [], []
    for row in rows[1:]:
        columns = row.find_all('td')
        if len(columns) == 7:
            date = columns[0].text
            actual_eps = columns[4].text
            estimated_eps = columns[3].text
            #list so info can be added
            earnings_dates.append(date)
            earnings_diff.append(float(actual_eps)-float(estimated_eps))

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

    return percent_momentum.fillna(method='bfill')


def get_historical_info() -> None:
    """
    This function gets the historical info for the given company symbols.
    
    It uses many functions from other modules to process
    historical data and run models on them.
    """
    period = 14
    start_date = '2015-01-01'
    end_date = '2023-06-23'
    for company_ticker in company_symbols:
        ticker = yf.Ticker(company_ticker)

        #_________________ GET Data______________________#
        # Retrieve historical data for the ticker using the `history()` method
        stock_data = ticker.history(start=start_date, end=end_date, interval="1d")
        if len(stock_data) == 0:
            raise ConnectionError("Failed to get stock data. Check your internet")

        #_________________MACD Data______________________#
        ema12 = stock_data['Close'].ewm(span=12).mean()
        ema26 = stock_data['Close'].ewm(span=26).mean()
        signal_line = ema12 - ema26
        signal_line = signal_line.ewm(span=9).mean()
        histogram = signal_line-signal_line
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

        #_________________12 and 26 day Ema flips______________________#
        flips = process_flips(ema12.values, ema26.values)

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
            'flips': flips,
            'supertrend1': super_trend1.values.tolist(),
            'supertrend2': super_trend2.values.tolist(),
            'supertrend3': super_trend3.values.tolist(),
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
            'earnings diff': earnings_diff
        }

        #_________________Scale them 0-1______________________#
        for key, values in converted_data.items():
            if key in excluded_values:
                continue
            min_val = min(values)
            max_val = max(values)
            if min_val == max_val:
                # Rare cases where nothing is indicated
                # Extreme indicators, 0/1 ussually.
                continue
            converted_data[key] = [(val - min_val) / (max_val - min_val) for val in values]
        with open(f'{company_ticker}/info.json', 'w') as json_file:
            json.dump(converted_data, json_file)


if __name__ == '__main__':
    get_historical_info()
