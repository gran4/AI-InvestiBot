import urllib.request
import ssl
import json

import yfinance as yf
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from typing import Optional, List
from datetime import datetime

company_symbols = ["AAPL"]


def get_earnings_history(company_ticker: str, context: Optional[ssl.SSLContext] = None) -> List[List[str]]:
    """
    Gets earning history of a company as a list.

    Args:
        company_ticker str: company to get info of
        context Optional[ssl certificate]: ssl certificate to use

    Warning::
        IF YOU GET ERROR:
            urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1002)>
        Go to Python3.6 folder (or whatever version of python you're using) > double click on "Install Certificates.command" file. :D
        NOTE: ON macOS go to Macintosh HD > AAPLications > Python3.6(or whatever version of python you're using) > double click on "Install Certificates.command" file. :D

    Warning:
        YOU are probibly looking to use get_corrected_earnings_history not this

    Returns:
        Tuple: of 2 Lists made of: Date, EPS_difference, respectivly
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
    rows = table.find_all('tr')

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
    """
    # Calculate the number of days between the date and the reference date
    return date_object - reference_date

def earnings_since_time(dates, start_date):
    date_object = datetime.strptime(start_date, "%Y-%m-%d")
    # Convert the datetime object back to a string in the desired format
    converted_date = date_object.strftime("%b %d, %Y")
    reference_date = datetime.strptime(converted_date, "%b %d, %Y")
    return [date_time_since_ref(date, reference_date) for date in dates]

def modify_earnings_dates(dates, start_date):
    temp = [datetime.strptime(date_string, "%b %d, %Y") for date_string in dates]
    return earnings_since_time(temp, start_date)



def get_liquidity_spikes(data, z_score_threshold=2.0, gradual=False) -> List:
    # Calculate the rolling average and standard deviation of bid volume
    window_size = 20
    rolling_average = data.rolling(window_size).mean()
    rolling_std = data.rolling(window_size).std()

    # Calculate Z-scores to identify abnormal bid volume spikes
    z_scores = (data - rolling_average) / rolling_std

    if gradual:
        abnormal_spikes = z_scores
    else:
        # Detect abnormal bid volume spikes
        abnormal_spikes = data[z_scores > z_score_threshold]
    return abnormal_spikes


def calculate_momentum_oscillator(data: pd.DataFrame, period: int=14) -> pd.Series:
    """
    Calculate the momentum oscillator for a given period.
    
    Arguments:
        data pandas.DataFrame: containing price data (e.g., close prices)
        period int: represents the period for calculating the momentum
    
    Returns:
        momentum_oscillator pandas.Series: representing the momentum oscillator values
    """
    # Calculate the momentum
    momentum = data.diff(period)
    
    # Calculate the mean of positive and negative momentum
    positive_momentum = momentum[momentum > 0].rolling(window=period).mean()
    negative_momentum = momentum[momentum < 0].rolling(window=period).mean()
    
    # Calculate the momentum oscillator
    momentum_oscillator = 100 * (positive_momentum / abs(negative_momentum))
    
    return momentum_oscillator


def getHistoricalInfo():
    start_date = '2015-01-01'
    end_date = '2023-06-12'
    stock_data = yf.download("AAPL", start=start_date, end=end_date)
    for company_ticker in company_symbols:
        ticker = yf.Ticker(company_ticker)

        # Retrieve historical data for the ticker using the `history()` method
        stock_data = ticker.history(start=start_date, end=end_date, interval="1d")
        #stock_data = yf.download(company_ticker, start=start_date, end=end_date, progress=False)

        # Calculate the MACD using pandas' rolling mean functions
        stock_data['12-day EMA'] = stock_data['Close'].ewm(span=12).mean()
        stock_data['26-day EMA'] = stock_data['Close'].ewm(span=26).mean()
        stock_data['MACD'] = stock_data['12-day EMA'] - stock_data['26-day EMA']
        stock_data['Signal Line'] = stock_data['MACD'].ewm(span=9).mean()
        stock_data['Histogram'] = stock_data['MACD']-stock_data['Signal Line']
        stock_data['200-day EMA'] = stock_data['Close'].ewm(span=200).mean()

        # Calculate the change in price (impulse) and momentum
        #Basically makes it impulse MACD
        stock_data['Change'] = stock_data['Close'].diff()
        stock_data['Momentum'] = stock_data['Change'].rolling(window=10).sum()  # Example: Using a 10-day rolling sum

        #For Reversal trading
        stock_data['gradual-liquidity spike'] = get_liquidity_spikes(stock_data['Volume'], gradual=True)
        stock_data['4-liquidity spike'] = get_liquidity_spikes(stock_data['Volume'])
        stock_data['momentum_oscillator'] = calculate_momentum_oscillator(stock_data['Close'])


        temp = []
        shortmore = None
        for short, mid in zip(stock_data['12-day EMA'].values, stock_data['26-day EMA'].values):
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
        stock_data['flips'] = temp


        earnings_dates, earnings_diff = get_earnings_history(company_ticker)
        #Do more in the model since
        #we do not know the start or end, yet

        dates = stock_data.index.strftime('%Y-%m-%d').tolist()
        converted_data = {
            'Dates': dates,
            '12-day EMA': stock_data['12-day EMA'].values.tolist(),
            '26-day EMA': stock_data['26-day EMA'].values.tolist(),
            'MACD': stock_data['MACD'].values.tolist(),
            'Signal Line': stock_data['Signal Line'].values.tolist(),
            'Histogram': stock_data['Histogram'].values.tolist(),
            '200-day EMA': stock_data['200-day EMA'].values.tolist(),
            'flips': stock_data['flips'].values.tolist(),
            'Momentum': stock_data['Momentum'].values.tolist(),
            'Change': stock_data['Change'].values.tolist(),
            'gradual-liquidity spike': stock_data['gradual-liquidity spike'].values.tolist(),
            '4-liquidity spike': stock_data['4-liquidity spike'].values.tolist(),
            'momentum_oscillator': stock_data['momentum_oscillator'].values.tolist(),
            'earnings dates': earnings_dates,
            'earnings diff': earnings_diff
        }
        with open(f"{company_ticker}/info.json", "w") as json_file:
            json.dump(converted_data, json_file)

if __name__ == '__main__':
    getHistoricalInfo()
