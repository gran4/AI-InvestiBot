import urllib.request, ssl, json

from bs4 import BeautifulSoup
from typing import Optional, List
from datetime import datetime
from Tradingfuncs import excluded_values, company_symbols, process_flips

import yfinance as yf
import numpy as np
import pandas as pd



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
        Tuple: of 2 Lists made of: Date and EPS_difference, respectivly
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


def get_liquidity_spikes(data, z_score_threshold=2.0, gradual=False) -> pd.Series:
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


def calculate_momentum_oscillator(data, period=14):
    """
    Calculates the momentum oscillator for the given data series.

    Args:
        data (pd.Series): Input data series.
        n (int): Number of periods for the oscillator calculation.

    Returns:
        pd.Series: Momentum oscillator values.
    """
    momentum = data.diff(period)  # Calculate the difference between current and n periods ago
    percent_momentum = (momentum / data.shift(period)) * 100  # Calculate momentum as a percentage

    return percent_momentum.fillna(method='bfill')


def convert_0to1(data: pd.Series):
    return (data - data.min()) / (data.max() - data.min())


def get_historical_info():
    period = 14
    start_date = '2015-01-01'
    end_date = '2023-06-13'
    stock_data = yf.download("AAPL", start=start_date, end=end_date)
    for company_ticker in company_symbols:
        ticker = yf.Ticker(company_ticker)

        #_________________ GET Data______________________#
        # Retrieve historical data for the ticker using the `history()` method
        stock_data = ticker.history(start=start_date, end=end_date, interval="1d")
        #stock_data = yf.download(company_ticker, start=start_date, end=end_date, progress=False)

        #_________________MACD Data______________________#
        stock_data['12-day EMA'] = stock_data['Close'].ewm(span=12).mean()
        stock_data['26-day EMA'] = stock_data['Close'].ewm(span=26).mean()
        stock_data['MACD'] = stock_data['12-day EMA'] - stock_data['26-day EMA']
        stock_data['Signal Line'] = stock_data['MACD'].ewm(span=9).mean()
        stock_data['Histogram'] = stock_data['MACD']-stock_data['Signal Line']
        stock_data['200-day EMA'] = stock_data['Close'].ewm(span=200).mean()

        #_________________Basically Impulse MACD______________________#
        stock_data['Change'] = stock_data['Close'].diff()
        stock_data['Change'].fillna(stock_data['Change'].iloc[1], inplace=True)
        stock_data['Momentum'] = stock_data['Change'].rolling(window=10, min_periods=1).sum()  # Example: Using a 10-day rolling sum
        stock_data['Momentum'].fillna(stock_data['Momentum'].iloc[1], inplace=True)

        #_________________Breakout Model______________________#
        stock_data["Change"] = stock_data["Close"].diff()  # Price change from the previous day
        stock_data["Gain"] = stock_data["Change"].apply(lambda x: x if x > 0 else 0)  # Positive changes
        stock_data["Loss"] = stock_data["Change"].apply(lambda x: abs(x) if x < 0 else 0)  # Negative changes
        stock_data["Avg Gain"] = stock_data["Gain"].rolling(window=14, min_periods=1).mean()  # 14-day average gain
        stock_data["Avg Loss"] = stock_data["Loss"].rolling(window=14, min_periods=1).mean()  # 14-day average loss

        stock_data["RS"] = stock_data["Avg Gain"] / stock_data["Avg Loss"]
        stock_data["RSI"] = 100 - (100 / (1 + stock_data["RS"]))
        stock_data["RSI"].fillna(stock_data["RSI"].iloc[1], inplace=True)

        volatility = stock_data['Close'].diff().abs()  # Calculate price volatility
        trama = stock_data['Close'].rolling(window=period).mean()  # Calculate the initial TRAMA with the specified period
        stock_data['TRAMA'] = trama + (volatility * 0.1)  # Adjust the TRAMA by adding 10% of the volatility

        #_________________Reversal______________________#
        stock_data['gradual-liquidity spike'] = get_liquidity_spikes(stock_data['Volume'], gradual=True)
        stock_data['3-liquidity spike'] = get_liquidity_spikes(stock_data['Volume'], z_score_threshold=4)
        stock_data['momentum_oscillator'] = calculate_momentum_oscillator(stock_data['Close'])



        #_________________Scale them 0-1______________________#
        for info in stock_data.keys():
            if info in excluded_values:
                continue
            stock_data[info] = convert_0to1(stock_data[info])


        #_________________12 and 26 day Ema flips______________________#
        stock_data['flips'] = process_flips(stock_data['12-day EMA'].values, stock_data['26-day EMA'].values)

        #earnings stuffs
        earnings_dates, earnings_diff = get_earnings_history(company_ticker)
        
        #Do more in the model since
        #we do not know the start or end, yet
        dates = stock_data.index.strftime('%Y-%m-%d').tolist()


        #_________________Process to json______________________#
        converted_data = {
            'Dates': dates,
            'Close': stock_data['Close'].values.tolist(),
            '12-day EMA': stock_data['12-day EMA'].values.tolist(),
            '26-day EMA': stock_data['26-day EMA'].values.tolist(),
            'MACD': stock_data['MACD'].values.tolist(),
            'Signal Line': stock_data['Signal Line'].values.tolist(),
            'Histogram': stock_data['Histogram'].values.tolist(),
            '200-day EMA': stock_data['200-day EMA'].values.tolist(),
            'flips': stock_data['flips'].values.tolist(),
            'Momentum': stock_data['Momentum'].values.tolist(),
            'Change': stock_data['Change'].values.tolist(),
            'RSI': stock_data['RSI'].values.tolist(),
            'TRAMA': stock_data['TRAMA'].values.tolist(),
            'gradual-liquidity spike': stock_data['gradual-liquidity spike'].values.tolist(),
            '3-liquidity spike': stock_data['3-liquidity spike'].values.tolist(),
            'momentum_oscillator': stock_data['momentum_oscillator'].values.tolist(),
            'earnings dates': earnings_dates,
            'earnings diff': earnings_diff
        }
        with open(f"{company_ticker}/info.json", "w") as json_file:
            json.dump(converted_data, json_file)

if __name__ == '__main__':
    get_historical_info()
