import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style
from mpl_finance import candlestick_ohlc
style.use('ggplot')

def stockdataplotter(df, plottitle=None):
    df['Adj. Close'].plot() if 'Adj. Close' in df else df['Close'].plot()
    plt.suptitle(plottitle,fontsize=16)
    plt.show()

def movavgplotter(df, days=[100], plottitle=None):
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    ax2.bar(df.index, df['Volume'])
    ax1.plot(df.index, df['Adj. Close']) if 'Adj. Close' in df else ax1.plot(df.index, df['Close'])

    for day in days:
        if 'Adj. Close' in df:
            df['ma'] = df['Adj. Close'].rolling(window=day, min_periods=0).mean()
            ax1.plot(df.index, df['ma'])
        else:
            df['ma'] = df['Close'].rolling(window=day, min_periods=0).mean()
            ax1.plot(df.index, df['ma'])
    plt.suptitle(plottitle,fontsize=16)
    plt.show()

def candlestickplotter(df, sampledays='10D', plottitle=None):
    if 'Adj. Close' in df:
        df_ohlc = df['Adj. Close'].resample(sampledays).ohlc()
    else:
        df_ohlc = df['Close'].resample(sampledays).ohlc()
    df_volume = df['Volume'].resample(sampledays).sum()

    df_ohlc.reset_index(inplace=True)
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    ax1.xaxis_date()

    candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
    plt.suptitle(plottitle,fontsize=16)
    plt.show()
