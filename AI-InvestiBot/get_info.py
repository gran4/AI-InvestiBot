"""
Name:
    getInfo.py

Purpose:
    This module provides utility functions that retrieve information for the
    stock bot. This includes getting the earnings history, liquidity spikes,
    and the stock price.

Author:
    Grant Yul Hur

"""
import math
import json
import os

from typing import Optional, List

import yfinance as yf
import numpy as np


from trading_funcs import (
    suggested_companies,
    find_best_number_of_years,
    process_flips,
    supertrends,
    kumo_cloud,
    get_liquidity_spikes,
    calculate_momentum_oscillator,
    get_earnings_history,
)


__all__ = (
    'get_historical_info'
)



def update_dynamic_tuning(company_ticker, stock_data) -> None:
    """
    This function gets the historical info for the given company symbols.
    
    It uses many functions from other modules to process
    historical data and run models on them.
    """
    relevant_years = find_best_number_of_years(company_ticker, stock_data=stock_data)
    t = len(stock_data)/260
    while t+2 < relevant_years:
        relevant_years -= 1
    num_days = int(math.log(relevant_years*6.0834 + 1) * 20)+20
    with open(f'Stocks/{company_ticker}/dynamic_tuning.json', 'w') as json_file:
        json.dump({
            'relevant_years': relevant_years,
            'num_days': num_days,
        }, json_file)


def update_info(company_ticker, stock_data) -> None:
    #_________________MACD Data______________________#
    ema12 = stock_data['Close'].ewm(span=12).mean()
    ema26 = stock_data['Close'].ewm(span=26).mean()
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
    volatility.iloc[0] = volatility.iloc[1]
    # Calculate the initial TRAMA with the specified period
    trama = stock_data['Close'].rolling(window=14, min_periods=1).mean()
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
        'supertrend1': super_trend1.tolist(),
        'supertrend2': super_trend2.tolist(),
        'supertrend3': super_trend3.tolist(),
        'kumo_cloud': kumo_status.tolist(),
        'Momentum': momentum.values.tolist(),
        'Change': change.values.tolist(),
        'RSI': relative_strength_index.values.tolist(),
        'TRAMA': trama.values.tolist(),
        'Volatility': volatility.values.tolist(),
        'Bollinger Middle': bollinger_middle.values.tolist(),
        'Above Bollinger': above_bollinger.astype(int).tolist(),
        'Bellow Bollinger': bellow_bollinger.astype(int).tolist(),
        'gradual-liquidity spike': gradual_liquidity_spike.values.tolist(),
        '3-liquidity spike': liquidity_spike3.values.tolist(),
        'momentum_oscillator': momentum_oscillator.values.tolist(),
        'earnings dates': earnings_dates,
        'earning diffs': earnings_diff
    }
    with open(f'Stocks/{company_ticker}/info.json', 'w') as json_file:
        json.dump(converted_data, json_file)


def get_historical_info(companys: Optional[List[str]]=None) -> None:
    """
    This function gets the historical info for the given company symbols.
    
    It uses many functions from other modules to process
    historical data and run models on them.
    """
    if not companys:# NOTE: weird global/local work around
        companys = suggested_companies
    for company_ticker in companys:
        ticker = yf.Ticker(company_ticker)
        #_________________ GET Data______________________#
        # Retrieve historical data for the ticker using the `history()` method
        stock_data = ticker.history(interval="1d", period='max')
        if len(stock_data) == 0:
            raise ConnectionError("Failed to get stock data. Check your internet")
        if not os.path.exists(f'Stocks/{company_ticker}'):
            os.makedirs(f'Stocks/{company_ticker}')

        print(company_ticker)
        #temp = supertrends(stock_data)

        update_dynamic_tuning(company_ticker, stock_data)
        update_info(company_ticker, stock_data)

if __name__ == '__main__':
    get_historical_info()
