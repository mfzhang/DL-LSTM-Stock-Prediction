import data.get_stockdata as gsd
import pickle
from plotting import makeplots

## Download stock data
#gsd.getandsave_idxstockdata('AEX',reload_tickers=True,update_stockdata=False)
#
## Import csv file as DataFrame
#heineken = gsd.get_singlestockdata('csv',['EURONEXT/','HEIA'],'AEX')
#
## Plot
#makeplots.stockdataplotter(heineken,'AMS:HEIA')
#
## Get tickers
#with open("AEXtickers.pickle", "rb") as index:
#    tickers = pickle.load(index)


test = gsd.get_singlestockdata('quandl',['EURONEXT/','HEIA'],'AEX', dates=['2008-01-01','2008-01-03'],update_stockdata=True)
