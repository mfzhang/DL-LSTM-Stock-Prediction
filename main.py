import data.get_stockdata as gsd

# run test

#gsd.getandsave_idxstockdata('AEX',reload_tickers=True,update_stockdata=False)

df = gsd.get_singlestockdata('csv',['EURONEXT/','HEIA'],'AEX')