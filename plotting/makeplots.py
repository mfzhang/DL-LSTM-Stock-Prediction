import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style
from mpl_finance import candlestick_ohlc
style.use('ggplot')

class Plotter(object):
    def __init__(self, df):
        self.df = df

    def stockdataplotter(self, plottitle=None):
        self.df['Adj. Close'].plot() if 'Adj. Close' in self.df else self.df['Close'].plot()
        plt.suptitle(plottitle,fontsize=16)
        plt.show()

    def movavgplotter(self, days=[100], plottitle=None):
        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
        ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
        ax2.bar(self.df.index, self.df['Volume'])
        ax1.plot(self.df.index, self.df['Adj. Close']) if 'Adj. Close' in self.df else ax1.plot(self.df.index, self.df['Close'])

        for day in days:
            if 'Adj. Close' in self.df:
                self.df['ma'] = self.df['Adj. Close'].rolling(window=day, min_periods=0).mean()
                ax1.plot(self.df.index, self.df['ma'])
            else:
                self.df['ma'] = self.df['Close'].rolling(window=day, min_periods=0).mean()
                ax1.plot(self.df.index, self.df['ma'])
#        plt.suptitle(plottitle,fontsize=16)
        plt.show()

    def candlestickplotter(self, sampledays='10D', plottitle=None):
        if 'Adj. Close' in self.df:
            df_ohlc = self.df['Adj. Close'].resample(sampledays).ohlc()
        else:
            df_ohlc = self.df['Close'].resample(sampledays).ohlc()
        df_volume = self.df['Volume'].resample(sampledays).sum()

        df_ohlc.reset_index(inplace=True)
        df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
        ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
        ax1.xaxis_date()

        candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
        ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
#        plt.suptitle(plottitle,fontsize=16)
        plt.show()

