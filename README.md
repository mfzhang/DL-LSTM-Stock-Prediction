# Deep Learning Project (TU Delft: CS4180)
The purpose of this project is to predict stock market values based on a Long Short-Term Memory (LSTM) deep learning model. Only based on previous stock data (no news events will be considered), this time series model will predict whether a certain stock will rise or fall in the near future. This algorithm could help traders during the decision when to buy, keep or sell their stocks. As basis, an existing algorithm will be used. This model will be extended using other input data and alternating the time series window. This way, we hope to achieve better results which can be used for longer periods of time (as we do not want to retrain our model every trading day).

## Requirements
- Python 3
- pandas_datareader 
- tensorflow (preferably on GPU)
- sklearn
- mpl_finance

## Purpose of the project
- Get new insights into deep learning.

## Credits
- This project is based on a tutorial of [Thushan Ganegedara](https://www.datacamp.com/community/tutorials/lstm-python-stock-market).
- Some elements of the www.pythonprogramming.net finance [tutorial](https://pythonprogramming.net/getting-stock-prices-python-programming-for-finance/) are also used.
- Team members: Daan Cuppen, Jeff Maes, Isaac Seminck, Benjamin De Bosscher and Kipras Paliušis.
