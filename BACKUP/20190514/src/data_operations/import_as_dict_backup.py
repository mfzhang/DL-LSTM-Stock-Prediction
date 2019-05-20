'''Import stock data from .csv to dict
'''

import os
import pandas as pd

def get_data(data_source, market):
    '''Definition which returns stock data

    Args:
        data_source (str): source to get data from
        market (str): certain index from which the underlying stocks will be used
    Returns:
        dict: if source git and market AEX
    Raises:
        ValueError: if source is not git
        CalueError: if market is not AEX
    '''
    stocks = {}
    if market == 'AEX':
        if data_source == 'git':
            files = os.listdir('data/'+market)
            for file in files:
                df = pd.read_csv('data/'+market+'/'+file, \
                      delimiter=',', usecols=['Date', 'Open', 'High', 'Low', \
                                             'Adj Close', 'Volume'])
                df = df.sort_values('Date')
                stocks[file[:-4]] = df
            print('Loaded data from the GitHub repository')
            return stocks
        else:
            raise ValueError('This source does not exist. Only git is an option.')
    else:
        raise ValueError('This market does not exist. Only AEX is an option.')
