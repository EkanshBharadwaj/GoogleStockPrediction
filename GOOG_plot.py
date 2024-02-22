import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Function to get the user's choice for prediction interval
def get_user_choice():
    options = ["1 day", "2 days", "3 days", "4 days", "5 days", "6 days", "1 week",
               "2 weeks", "1 month", "2 months", "3 months", "6 months"]
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    choice = int(input("Enter your choice (1-12): "))
    return choice

# Function to calculate the prediction interval in days
def calculate_prediction_interval(choice):
    intervals = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 14, 9: 30, 10: 60, 11: 90, 12: 180}
    return intervals.get(choice, 1)  # Default to 1 day if invalid choice

# Load the saved LSTM model
model_path = 'stock_prediction_model.h5'
model = load_model(model_path)

# Load the original CSV file to get the actual data
stock_data = pd.read_csv('GOOG_stock_data_2000_2024.csv')
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.set_index('Date', inplace=True)

# Normalize the 'Close' prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

# Get the user's choice for the prediction interval
user_choice = get_user_choice()
prediction_interval = calculate_prediction_interval(user_choice)

# Define the number of days to look back
look_back = 60  # This should match the look_back used during training

# Prepare the last 'look_back' days of data as input for the model
input_data = scaled_data[-look_back:]
input_data = np.reshape(input_data, (1, input_data.shape[0], 1))

# Predict the future prices for the prediction interval
predicted_prices = []
current_batch = input_data
for i in range(prediction_interval):
    # Get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]
    
    # Append the prediction into the array
    predicted_prices.append(current_pred)
    
    # Use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)

predicted_prices = scaler.inverse_transform(predicted_prices)

# Prepare the data for plotting
historical_data = stock_data['Close'].iloc[-look_back:]
historical_dates = historical_data.index

# Generate dates for prediction, starting after the last historical date
last_historical_date = historical_dates[-1]
predicted_dates = pd.date_range(start=last_historical_date, periods=prediction_interval + 1, freq='B')[1:]

# Ensure the length of predicted prices matches the length of predicted dates
predicted_prices = predicted_prices.flatten()[:len(predicted_dates)]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(historical_dates, historical_data, label='Historical Data', color='blue')

# To connect the lines, start predicted plot from the last actual data point
last_actual_price = historical_data.iloc[-1]
connected_predicted_prices = np.insert(predicted_prices, 0, last_actual_price)

# Adjust connected_predicted_dates to include the last actual date for a seamless connection
connected_predicted_dates = pd.date_range(start=historical_dates[-1], periods=len(connected_predicted_prices), freq='B')

# Now plot with the connected data
plt.plot(connected_predicted_dates, connected_predicted_prices, label='Predicted', color='red')

# Formatting the x-axis to show dates
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Adjust the number of x-ticks shown
plt.xticks(rotation=45)  # Rotate the dates for better readability

plt.legend()
plt.title(f"Stock Prices: Last {look_back} Days and Next {len(predicted_dates)} Predicted Days")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
plt.show()