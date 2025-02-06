import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import messagebox, scrolledtext

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
    data = data.dropna().copy()
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
    
    return f"{ticker}: Latest Close: {latest_close:.2f}, Predicted Price: {prediction:.2f}, Signal: {signal}\n"

def run_analysis():
    tickers = entry.get().split(',')
    output_text.delete('1.0', tk.END)  # Clear previous output
    for ticker in tickers:
        result = evaluate_stock(ticker.strip())
        output_text.insert(tk.END, result)

# GUI Setup
root = tk.Tk()
root.title("Stock Analysis")

tk.Label(root, text="Enter Stock Tickers (comma-separated):").pack()
entry = tk.Entry(root, width=50)
entry.pack()

tk.Button(root, text="Analyze", command=run_analysis).pack()

output_text = scrolledtext.ScrolledText(root, width=60, height=10)
output_text.pack()

root.mainloop()