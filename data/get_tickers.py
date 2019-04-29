''' Automatically acquire underlying stock tickers from certain index
        Code based on tutorial of pythonprogramming.net
'''

import pickle
import requests
import bs4 as bs

def save_tickers(idx):
    '''Definition which returns list of symbol which comes before ticker
            when acquiring dataset followed by tickers of a certain index
    Args:
        idx (str): index symbol
    Returns:
        tickers (lst): ['symbol which comes before ticker when acquiring dataset','tickers']
    Raises:
        ValueError: if idx is not equal to SP500, BEL20, AEX
    '''
    if idx == 'SP500':
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = ['WIKI/']
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            tickers.append(ticker)

        with open("SP500tickers.pickle", "wb") as SP500:
            pickle.dump(tickers, SP500)

        return tickers

    elif idx == 'BEL20':
        resp = requests.get('https://en.wikipedia.org/wiki/BEL_20')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = ['EURONEXT/']
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[2].text
            tickers.append(ticker)

        with open("BEL20tickers.pickle", "wb") as BEL20:
            pickle.dump(tickers, BEL20)

        return tickers

    elif idx == 'AEX':
        resp = requests.get('https://en.wikipedia.org/wiki/AEX_index')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = ['EURONEXT/']
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[2].text
            tickers.append(ticker)

        with open("AEXtickers.pickle", "wb") as AEX:
            pickle.dump(tickers, AEX)

        return tickers

    else:
        raise ValueError('This idx is not an option')
