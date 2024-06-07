import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


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
    data['RSI'] = RSIIndicator(data['Close'], window=14).rsi().fillna(0)
    macd = MACD(data['Close'])
    data['MACD'] = macd.macd().fillna(0)
    data['MACD_Signal'] = macd.macd_signal().fillna(0)
    bollinger = BollingerBands(data['Close'])
    data['BB_High'] = bollinger.bollinger_hband().fillna(0)
    data['BB_Low'] = bollinger.bollinger_lband().fillna(0)

    # Calculate Moving Average
    data['50_MA'] = data['Close'].rolling(window=50).mean().fillna(0)

    # Add benchmark closing prices as a new feature
    data['Benchmark_Close'] = benchmark_data['Close']

    # Fill remaining NaN values
    data.fillna(0, inplace=True)

    # Step 2: Data Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label=stock_ticker)
    plt.title(f'Historical Closing Prices of {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Step 3: Risk Analysis
    returns = data['Close'].pct_change()
    volatility = returns.std()
    annual_volatility = volatility * np.sqrt(252)
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

    print(f"Annualized Volatility of {stock_ticker}: {annual_volatility}")
    print(f"Sharpe Ratio of {stock_ticker}: {sharpe_ratio}")

    # Step 4: Prepare Data for Gradient Boosted Trees
    features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'Rate of Return', '50_MA',
                'Benchmark_Close']
    X = data[features].values
    y = data['Close'].shift(-1).fillna(method='ffill').values  # Predict next day's closing price

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Splitting data into train and test sets
    train_size = int(len(X) * (1 - test_ratio))
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Step 5: Hyperparameter Tuning with Grid Search
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt', 'log2'],
        'alpha': [0.1, 0.5]  # Adjusted 'alpha' parameter range
    }

    grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), param_grid=param_grid, cv=3,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Step 6: Train the Model with Best Parameters
    best_model = GradientBoostingRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    # Step 7: Make Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Step 8: Visualization
    plt.figure(figsize=(14, 7))

    plt.plot(data.index[:train_size], y_train, label='Actual Train Prices')
    plt.plot(data.index[train_size:], y_test, label='Actual Test Prices')
    plt.plot(data.index[:train_size], y_pred_train, label='Predicted Train Prices')
    plt.plot(data.index[train_size:], y_pred_test, label='Predicted Test Prices')

    plt.title(f'Stock Price Prediction of {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# Example usage:
stock_market_analysis('NVDA', test_ratio=0.2)
