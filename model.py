import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Load the dataset
stock_data = pd.read_csv(r'C:\Users\ekans\Desktop\Stock Market Prediction\GOOG_stock_data_2000_2024.csv')
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.set_index('Date', inplace=True)

# Define the outlier periods
outlier_periods = [
    (pd.Timestamp('2004-08-19'), pd.Timestamp('2006-01-01')), # Possible IPO fluctuations
    (pd.Timestamp('2007-01-01'), pd.Timestamp('2009-02-01')),  # The 2008 financial crisis
    (pd.Timestamp('2020-01-01'), pd.Timestamp('2021-06-01'))   # The COVID-19 pandemic
]

# Remove the outlier periods from the dataset
for start, end in outlier_periods:
    stock_data = stock_data[(stock_data.index < start) | (stock_data.index > end)]

# Use only the 'Close' prices
close_prices = stock_data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Define the dataset creation function
def create_dataset(dataset, look_back=60):# was 1

    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60  # Use 60 days of past data to predict the next day
X, Y = create_dataset(scaled_data, look_back)

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Model path
model_path = r'C:\Users\ekans\Desktop\Stock Market Prediction\stock_prediction_model.h5'

# Check if the model already exists
if not os.path.exists(model_path):
    # Create and fit the LSTM network with increased complexity
    model = Sequential()
    model.add(Input(shape=(look_back, 1)))
    model.add(LSTM(units=100, return_sequences=True))  # Increased number of units
    model.add(Dropout(0.2))  # Added dropout for regularization
    model.add(LSTM(units=100, return_sequences=True))  # Another LSTM layer
    model.add(Dropout(0.2))  # Added dropout for regularization
    model.add(LSTM(units=100))  # Yet another LSTM layer

    
    model.add(Dense(units=1))  # Output layer

    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Early stopping callback to prevent overfitting
    #early_stopping = EarlyStopping(monitor='val_loss', patience=75, restore_best_weights=True)
    
    # Fit the model with a validation split for early stopping
    model.fit(X_train, Y_train, epochs=1000, batch_size=32, verbose=2, validation_split=0.1) # old model was with 300 epochs callbacks=[early_stopping]

    # Save the trained model
    model.save(model_path)
else:
    # Load the existing model
    model = load_model(model_path)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions back to normal values
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Calculate root mean squared error
train_score = np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))
print(f'Train Score: {train_score:.2f} RMSE')
test_score = np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))
print(f'Test Score: {test_score:.2f} RM')