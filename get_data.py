import requests
import pandas as pd
import time
from datetime import datetime
import pandas_datareader.data as web


def historical_stooq(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical stock data from Stooq, fixed interval is 1d.
    """
    start_date = int(time.mktime(datetime.strptime(start_date, '%Y-%m-%d').timetuple()))
    end_date = int(time.mktime(datetime.strptime(end_date, '%Y-%m-%d').timetuple()))
    data = web.DataReader(symbol, "stooq", start_date, end_date)
    return data

def historical_yahoo(symbol: str, start_date: str, end_date: str, interval: str='1d') -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance, minimum interval is 1m (only 8 days period can be downloaded).
    Recommended interval is 1d, 1h, 15m, 5m, 1m.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    start_date = int(time.mktime(datetime.strptime(start_date, '%Y-%m-%d').timetuple()))
    end_date = int(time.mktime(datetime.strptime(end_date, '%Y-%m-%d').timetuple()))

    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_date}&period2={end_date}&interval={interval}'
    response = requests.get(url, headers=headers)

    if response.json()['chart']['error'] is not None:
        raise Exception(f'Failed to get stock data: Error code {response.status_code}: {response.json()["chart"]["error"]["description"]}')

    data = response.json()['chart']['result'][0]
    quote = data['indicators']['quote'][0]
    date = pd.to_datetime(data['timestamp'], unit='s')

    # if interval is 1d
    if 'd' in interval:
        date = date.normalize()
        adj_close = data['indicators']['adjclose'][0]

        stock_data = pd.DataFrame({
            'Date': date,
            'Open': quote['open'],
            'High': quote['high'],
            'Low': quote['low'],
            'Close': quote['close'],
            'Volume': quote['volume'],
            'Adj_close': adj_close['adjclose']
        })
    
    # if interval is 1h, 1m, 15m, 5m
    elif 'h' in interval or 'm' in interval:
        stock_data = pd.DataFrame({
            'Date': date,
            'Open': quote['open'],
            'High': quote['high'],
            'Low': quote['low'],
            'Close': quote['close'],
            'Volume': quote['volume'],
        })
        
    stock_data.set_index('Date', inplace=True)
    
    return stock_data

if __name__ == '__main__':
    print(historical_yahoo('AAPL', '2025-12-22', '2026-01-01', '1d'))
