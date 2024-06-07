#READ ME: This one not compiling

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional


def create_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def stock_market_analysis(stock_ticker, start_date="2010-01-01", end_date="2024-01-01", test_ratio=0.2,
                          benchmark_ticker='VOO'):
    # Step 1: Retrieve Stock Data
    data = yf.download(stock_ticker, start=start_date, end=end_date)
    benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)

    if data.empty or benchmark_data.empty:
        print("Failed to retrieve data. Check the ticker symbols and network connection.")
        return

    # Fill NaN values with 0
    data['Volume'] = data['Volume'].fillna(0)
    benchmark_data['Volume'] = benchmark_data['Volume'].fillna(0)

    # Calculate additional features
    data['Rate of Return'] = data['Close'].pct_change().fillna(0)
    data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
    macd = MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD Signal'] = macd.macd_signal()
    bb = BollingerBands(data['Close'])
    data['BB High'] = bb.bollinger_hband()
    data['BB Low'] = bb.bollinger_lband()

    # Step 2: Preprocess Data
    data = data.dropna()
    features = ['Close', 'RSI', 'MACD', 'MACD Signal', 'BB High', 'BB Low', 'Volume']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    x = []
    y = []
    window_size = 60
    for i in range(window_size, len(scaled_data)):
        x.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)
    test_size = int(test_ratio * len(x))
    x_train, x_test = x[:-test_size], x[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    # Step 3: Build and Train Model
    input_shape = (x_train.shape[1], x_train.shape[2])

    def build_model(optimizer='adam'):
        model = create_model(input_shape)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    keras_reg = KerasRegressor(build_fn=build_model, verbose=0)

    param_distribs = {
        "batch_size": Integer(10, 128),
        "optimizer": ["adam", "rmsprop"],
        "epochs": Integer(10, 100),
    }

    bayes_search = BayesSearchCV(estimator=keras_reg, search_spaces=param_distribs, n_iter=10, cv=3, n_jobs=-1)

    bayes_search.fit(x_train, y_train)

    print("Best parameters found: ", bayes_search.best_params_)
    print("Best validation score: ", bayes_search.best_score_)


# Example usage:
stock_market_analysis('NVDA')
