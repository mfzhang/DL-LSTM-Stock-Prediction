import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc

def adjclose(df, plottitle=None):
    df['Adj Close'].plot() if 'Adj Close' in df else df['Close'].plot()
    plt.suptitle(plottitle,fontsize=16)
    plt.show()

def movavg(df, days=[100], plottitle=None):
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

def candlestick(df, sampledays='10D', plottitle=None):
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

def prediction(df, pp_data, x_axis_seq, predictions_over_time, best_prediction_epoch):
    plt.figure(figsize = (18,18))
    plt.subplot(2,1,1)
    plt.plot(range(df.shape[0]),pp_data.all_mid_data,color='b')
    
    # Plotting how the predictions change over time
    # Plot older predictions with low alpha and newer predictions with high alpha
    start_alpha = 0.25
    alpha  = np.arange(start_alpha,1.1,(1.0-start_alpha)/len(predictions_over_time[::3]))
    for p_i,p in enumerate(predictions_over_time[::3]):
        for xval,yval in zip(x_axis_seq,p):
            plt.plot(xval,yval,color='r',alpha=alpha[p_i])
    
    plt.title('Evolution of Test Predictions Over Time',fontsize=16)
    plt.xlabel('Date',fontsize=16)
    plt.ylabel('Mid Price',fontsize=16)
    plt.xlim(left=pp_data.split_datapoint)
    
    plt.subplot(2,1,2)
    
    # Predicting the best test prediction you got
    plt.plot(range(df.shape[0]),pp_data.all_mid_data,color='b')
    for xval,yval in zip(x_axis_seq,predictions_over_time[best_prediction_epoch]):
        plt.plot(xval,yval,color='r')
        
    plt.title('Best Test Predictions Over Time',fontsize=16)
    plt.xlabel('Date',fontsize=16)
    plt.ylabel('Mid Price',fontsize=16)
    plt.xlim(left=pp_data.split_datapoint)
    plt.show()
    plt.savefig('plots/last_prediction.pdf')
