import yfinance as yf
import pandas as pd
import matplotlib as mpl
import numpy as np
import Keras as keras


def goog_data_filtered():
    # Download stock data
    #Note: It will only  show data from August 19, 2004 (Google's IPO date)
    stock_data = yf.download("GOOG", period= "max", auto_adjust=True)
    # Resetting the index to make 'Date' a column
    stock_data.reset_index(inplace=True)
    # Selecting only the required columns
    filtered_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close']]
    # Specify the file path
    file_path = r'C:\Users\ekans\Desktop\Stock Market Prediction\GOOG_stock_data_2000_2024.csv'
    # Save the filtered data to a CSV file
    filtered_data.to_csv(file_path, index=False)

goog_data_filtered()