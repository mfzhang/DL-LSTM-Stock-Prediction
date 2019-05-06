''' Download stock data using Quandl API
'''

import os
import pickle
import datetime as dt
import pandas as pd
import quandl
import data.get_tickers as gt

def get_singlestockdata(source, stockinput, idx='no_index', dates=[None, None], update_stockdata=False):
    '''Definition which returns single stock data

    Args:
        stockinput (lst of len 2): ['symbol which comes before ticker when acquiring dataset','ticker']
        idx (optional[str]): index symbol
        dates (optional[lst of len 2]): [dt.date(startdate), dt.date(enddate)]
        update_stockdata (optional[bool]): update stock data if it already exists
    Returns:
        pd.DataFrame: if stock data not exists
        pd.DataFrame: if stock data exists and update_stockdata=True
        None: if stock data exists and update_stockdata=False
    Raises:
        ValueError: if source is not csv or quandl
    '''
    if source == 'csv':
        df = pd.read_csv('data/stock_data/'+idx+'/'+stockinput[1]+'.csv', parse_dates=True, index_col=0)
        df.rename(columns={'Last':'Close'}, inplace=True)
    elif source == 'quandl':
        if update_stockdata:
            print('Downloading {} from {}'.format(stockinput[1], idx))
            df = quandl.get(stockinput[0]+stockinput[1], start_date=dates[0], end_date=dates[1], authtoken="ruJzH3a2GZ3PHtneDSoZ", paginate=True)
            df.rename(columns={'Last':'Close'}, inplace=True)
        else:
            if not os.path.exists('data/stock_data/{}/{}.csv'.format(idx, stockinput[1])):
                print('Downloading {} from {}'.format(stockinput[1], idx))
                df = quandl.get(stockinput[0]+stockinput[1], start_date=dates[0], end_date=dates[1], authtoken="ruJzH3a2GZ3PHtneDSoZ", paginate=True)
                df.rename(columns={'Last':'Close'}, inplace=True)
            else:
                df = None
                print('Already downloaded {} from {}'.format(stockinput[1], idx))
    else:
        raise ValueError('This source is not an option')

    return df

def save_singlestockdata(df, ticker, idx='no_index'):
    '''Definition which saves pd.DataFrame as a csv file

    Args:
        df (pd.DataFrame): stock data which should be saved
        ticker (str): ticker of stock
        idx (optional[str]): index symbol
    Returns:
        None
    '''
#    df.rename(columns={'Adj Close':ticker}, inplace=True)
#
#    if not os.path.exists('data/stock_data/{}'.format(idx)):
#        os.makedirs('data/stock_data/{}'.format(idx))
#
#    df.to_csv('data/stock_data/{}/{}.csv'.format(idx, ticker))
    raise ValueError('Already downloaded {} from {}'.format(ticker, idx))

def getandsave_idxstockdata(idx, dates=[None, None], reload_tickers=False, update_stockdata=False):
    '''Definition which gets all stock data of a certain index

    Args:
        idx (str): index symbol
        dates (optional[lst of len 2]): [dt.date(startdate), dt.date(enddate)]
        reload_tickers (optional[bool]): update tickers if .pickle already exists
        update_stockdata (optional[bool]): update stock data if it already exists
    Returns:
        None
    '''
    if reload_tickers:
        tickers = gt.save_tickers(idx)
    else:
        try:
            with open("{}tickers.pickle".format(idx), "rb") as index:
                tickers = pickle.load(index)
        except:
            tickers = gt.save_tickers(idx)
    for ticker in tickers[1:]:
        try:
            df = get_singlestockdata('quandl', [tickers[0], ticker], idx, dates, update_stockdata)
        except:
            print(ticker+' skipped because not found on source')
            df = None
        if type(df) != type(None):
            save_singlestockdata(df, ticker, idx)
