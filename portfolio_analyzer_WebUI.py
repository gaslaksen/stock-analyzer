import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import time

# Function to fetch stock data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1y')
    return hist

# Function to train LSTM model and make predictions
def lstm_predict(data):
    if len(data) < 60:  # Ensure enough data for LSTM training
        return np.nan  # Not enough data for meaningful prediction
    
    data = data[['Close']].copy()
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    # Prepare training data
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):  # Use 60-day window
        X_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

    # Make prediction
    last_60_days = scaled_data[-60:]  # Use last 60 days for prediction
    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
    predicted_price = model.predict(last_60_days)
    predicted_price = scaler.inverse_transform(predicted_price)

    return predicted_price[0][0]

# Function to evaluate a stock
def evaluate_stock(ticker):
    data = fetch_stock_data(ticker)
    if data.empty:
        return None, None
    
    prediction = lstm_predict(data)
    latest_close = data['Close'].iloc[-1]
    signal = 'Buy' if prediction > latest_close else 'Sell'

    return {
        'Ticker': ticker,
        'Latest Close': latest_close,
        'Predicted Price': prediction,
        'Signal': signal
    }, data

# Function to update stock prices in real-time
def refresh_data(tickers):
    results = []
    for ticker in tickers:
        ticker = ticker.strip().upper()
        stock_result, stock_data = evaluate_stock(ticker)

        if stock_result:
            results.append(stock_result)

            # Plot chart
            st.subheader(f"{ticker} - Stock Price Chart")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(stock_data.index, stock_data['Close'], label="Close Price", color='blue')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

    # Display results in a table
    if results:
        results_df = pd.DataFrame(results)
        st.write(results_df)

# Streamlit UI
st.title("📊 Real-Time Stock Prediction with LSTM AI")

tickers_input = st.text_input("Enter Stock Tickers (comma-separated)", "AAPL, GOOGL, MSFT")
refresh_interval = st.slider("Refresh Interval (seconds)", min_value=30, max_value=300, value=60)

# Create a refresh button
if st.button("Refresh Now"):
    refresh_data(tickers_input.split(','))

# Automatic updates every X seconds
st.write(f"Auto-refreshing every {refresh_interval} seconds...")
while True:
    time.sleep(refresh_interval)
    refresh_data(tickers_input.split(','))