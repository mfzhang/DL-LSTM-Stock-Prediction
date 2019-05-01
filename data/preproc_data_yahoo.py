import pandas as pd

def yahoo(ticker, idx='AEX'):
    
    df = pd.read_csv('stock_data/'+idx+'/'+ticker+'.AS'+'.csv', parse_dates=True, index_col=0)
    df = df.dropna()
    df.to_csv('stock_data/{}/{}.csv'.format(idx, ticker))

