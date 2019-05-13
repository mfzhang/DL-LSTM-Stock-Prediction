'''Preprocess data before applying algorithm
        First, test and train data should be splitted
        Then, the data will be normalized
        Lastly, exponential moving average smoothing will be applied
'''

import numpy as np
from sklearn.preprocessing import MinMaxScaler

class PreProc(object):
    '''Preprocessing object
            TO BE COMPLETED
    '''
    def __init__(self, df):
        self.df = df
        self.high_prices = df.loc[:, 'High'].values
        self.low_prices = df.loc[:, 'Low'].values
        self.mid_prices = (self.high_prices+self.low_prices)/2.0
        self.scaler = MinMaxScaler()
        self.train_data = 0
        self.test_data = 0
        self.all_mid_data = 0
        self.split_datapoint = 0

    def splitdata(self, split_datapoint):
        '''Method which splits test data and train data
        '''
        self.split_datapoint = split_datapoint
        self.train_data = self.mid_prices[:split_datapoint].reshape(-1, 1)
        self.test_data = self.mid_prices[split_datapoint:].reshape(-1, 1)

    def normalize_smooth(self, smoothing_window_size, EMA=0.0, gamma=0.1):
        '''Normalizes and smooths training data (and test data)
        '''
        # Train the Scaler with training data and smooth data
        for di in range(0, self.split_datapoint-smoothing_window_size, smoothing_window_size):  # FIX RANGES, DUNNO YET
            self.scaler.fit(self.train_data[di:di+smoothing_window_size, :])
            self.train_data[di:di+smoothing_window_size, :] = \
                self.scaler.transform(self.train_data[di:di+smoothing_window_size, :])

        # You normalize the last bit of remaining data
        self.scaler.fit(self.train_data[di+smoothing_window_size:, :])
        self.train_data[di+smoothing_window_size:, :] = \
            self.scaler.transform(self.train_data[di+smoothing_window_size:, :])

        # Reshape both train and test data
        self.train_data = self.train_data.reshape(-1)

        # Normalize test data
        self.test_data = self.scaler.transform(self.test_data).reshape(-1)

        # Now perform exponential moving average smoothing
        # So the data will have a smoother curve than the original ragged data
        for ti in range(self.split_datapoint):
            EMA = gamma*self.train_data[ti] + (1-gamma)*EMA
            self.train_data[ti] = EMA

            # Used for visualization and test purposes
            self.all_mid_data = np.concatenate([self.train_data, self.test_data], axis=0)
