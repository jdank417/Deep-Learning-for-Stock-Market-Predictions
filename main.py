# "Simple" Implementation of Stock Market Analysis using LSTM Neural Networks. Used for reference in the main.py file.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import os

def create_simple_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 5)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dense(1))  # Predicting only the 'Close' price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def stock_market_analysis(stock_ticker, start_date="2010-01-01", end_date="2024-01-01", test_ratio=0.2, benchmark_ticker='VOO'):
    # Retrieve Stock Data
    data = yf.download(stock_ticker, start=start_date, end=end_date)
    benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)

    if data.empty or benchmark_data.empty:
        print("Failed to retrieve data. Check the ticker symbols and network connection.")
        return

    # Fill NaN values with forward fill method
    data.fillna(method='ffill', inplace=True)
    benchmark_data.fillna(method='ffill', inplace=True)

    # Calculate additional features
    data['Rate of Return'] = data['Close'].pct_change().fillna(0)
    data['RSI'] = RSIIndicator(data['Close'], window=14).rsi().fillna(0)
    data['50_MA'] = SMAIndicator(data['Close'], window=50).sma_indicator().fillna(0)

    # Fill remaining NaN values
    data.fillna(0, inplace=True)

    # Data Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label=stock_ticker)
    plt.title(f'Historical Closing Prices of {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Risk Analysis
    returns = data['Close'].pct_change()
    volatility = returns.std()
    annual_volatility = volatility * np.sqrt(252)
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

    print(f"Annualized Volatility of {stock_ticker}: {annual_volatility}")
    print(f"Sharpe Ratio of {stock_ticker}: {sharpe_ratio}")

    # Prepare Data for LSTM
    features = ['Close', 'Volume', 'Rate of Return', 'RSI', '50_MA']
    stock_data = data[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)

    def create_dataset(data, time_step=1):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), :])
            Y.append(data[i + time_step, 0])  # Target is the 'Close' price
        return np.array(X), np.array(Y)

    time_step = 60
    x, y = create_dataset(scaled_data, time_step)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, shuffle=False)

    # Create and Train the Model
    model = create_simple_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, mode='min')

    model_dir = f"{stock_ticker}_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    checkpoint_path = os.path.join(model_dir, "best_model.keras")
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.2, callbacks=[early_stopping, model_checkpoint, reduce_lr])

    # Save the final model architecture
    model_json = model.to_json()
    with open(os.path.join(model_dir, "final_model.json"), "w") as json_file:
        json_file.write(model_json)

    # Make Predictions and Plot Results
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]

    prediction_start_index = len(data) - len(predictions)
    prediction_dates = data.index[prediction_start_index:]

    valid = pd.DataFrame(index=prediction_dates, data=predictions, columns=['Predictions_Close'])

    plt.figure(figsize=(16, 7))
    plt.plot(data['Close'][:len(y_train) + time_step], label='Actual Prices (Train)')
    plt.plot(data['Close'][len(y_train) + time_step:], label='Actual Prices (Test)')
    plt.plot(valid.index, valid['Predictions_Close'], label='Predictions (Close)', color='green')
    plt.title(f'Stock Price Prediction of {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Example usage:
stock_market_analysis('NVDA', test_ratio=0.5)
