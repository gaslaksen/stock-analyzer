import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1y')
    return hist

def calculate_technical_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(14).mean()))
    return data

def ai_based_prediction(data):
    data = data.dropna().copy()  # Ensure we are working with a copy of the data
    data['Days'] = np.arange(len(data)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(data[['Days']], data['Close'].values.reshape(-1, 1))
    future = pd.DataFrame({'Days': [len(data)]})
    prediction = model.predict(future)
    return prediction[0][0]

def evaluate_stock(ticker):
    data = fetch_stock_data(ticker)
    data = calculate_technical_indicators(data)
    prediction = ai_based_prediction(data)
    latest_close = data['Close'].iloc[-1]
    signal = 'Buy' if prediction > latest_close else 'Sell'
    
    return {
        'Ticker': ticker,
        'Latest Close': latest_close,
        'Predicted Price': prediction,
        'Signal': signal
    }

def main():
    tickers = input("Enter stock tickers separated by commas: ").split(',')
    results = [evaluate_stock(ticker.strip()) for ticker in tickers]
    results_df = pd.DataFrame(results)
    print(results_df)

if __name__ == "__main__":
    main()