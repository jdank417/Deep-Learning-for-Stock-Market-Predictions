import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import logging

# Configure logging
logging.basicConfig(filename='stock_prediction.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stock_prediction')


def create_model(lstm_units=100, conv_filters=64, conv_kernel_size=3, dropout_rate=0.3, time_steps=60, num_features=7):
    model = Sequential([
        Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu',
               input_shape=(time_steps, num_features)),
        MaxPooling1D(pool_size=2),
        LSTM(units=lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=lstm_units),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Changed to linear activation
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def add_technical_indicators(data):
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'] = calculate_macd(data['Close'])
    data['ATR'] = calculate_atr(data)
    data.dropna(inplace=True)
    return data


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line


def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()


def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def stock_market_analysis(stock_symbol, test_ratio, future_days):
    logger.info(f"Starting analysis for stock symbol: {stock_symbol}")

    # Download stock data
    stock_data = yf.download(stock_symbol, start='2010-01-01', end='2024-01-01')
    logger.info(f"Downloaded data shape: {stock_data.shape}")

    # Add technical indicators
    stock_data = add_technical_indicators(stock_data)
    logger.info(f"Data shape after adding indicators: {stock_data.shape}")

    # Prepare features and target
    features = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'MACD', 'ATR']
    data = stock_data[features].values

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create a separate scaler for 'Close' price only
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(data[:, 0].reshape(-1, 1))

    # Prepare the dataset
    time_steps = 60
    X, y = create_dataset(scaled_data, time_steps)

    # Split the data
    train_size = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create and train the model
    model = create_model(time_steps=time_steps, num_features=len(features))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
    ]

    # Use TimeSeriesSplit for validation
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, val_index in tscv.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold),
                  epochs=100, batch_size=32, callbacks=callbacks, verbose=0)

    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform predictions and actual values using close_scaler
    predictions = close_scaler.inverse_transform(predictions)
    actual_prices = close_scaler.inverse_transform(scaled_data[:, 0].reshape(-1, 1))

    # Prepare for plotting
    all_predictions = np.full(len(actual_prices), np.nan)
    all_predictions[-len(predictions):] = predictions.flatten()

    # Create a directory for saving plots
    plot_dir = 'stock_plots'
    os.makedirs(plot_dir, exist_ok=True)

    # Plotting historical and predicted prices
    plt.figure(figsize=(16, 8))

    # Plot actual prices
    plt.plot(stock_data.index, actual_prices, label='Actual Prices', color='blue')

    # Plot predicted prices
    valid_predictions = ~np.isnan(all_predictions)
    plt.plot(stock_data.index[valid_predictions], all_predictions[valid_predictions], label='Predicted Prices',
             color='red')

    plt.title(f'{stock_symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add text annotations for min and max values
    min_price = np.min(actual_prices)
    max_price = np.max(actual_prices)
    plt.annotate(f'Min: {min_price:.2f}', xy=(stock_data.index[np.argmin(actual_prices)], min_price), xytext=(10, 10),
                 textcoords='offset points', ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.annotate(f'Max: {max_price:.2f}', xy=(stock_data.index[np.argmax(actual_prices)], max_price), xytext=(10, -10),
                 textcoords='offset points', ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.savefig(os.path.join(plot_dir, f'{stock_symbol}_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Prediction plot saved for {stock_symbol}")

    # Future predictions
    last_sequence = scaled_data[-time_steps:]
    future_predictions = []
    for _ in range(future_days):
        next_pred = model.predict(last_sequence.reshape(1, time_steps, len(features)))
        future_predictions.append(next_pred[0, 0])
        next_pred_full = last_sequence[-1].copy()
        next_pred_full[0] = next_pred[0, 0]
        last_sequence = np.append(last_sequence[1:], [next_pred_full], axis=0)

    future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=future_days)
    future_predictions = close_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Plotting future predictions
    plt.figure(figsize=(16, 8))

    # Plot historical prices
    plt.plot(stock_data.index, actual_prices, label='Historical Prices', color='blue')

    # Plot future predictions
    plt.plot(future_dates, future_predictions, label='Future Predictions', color='red')

    plt.title(f'{stock_symbol} Stock Price Future Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add text annotations for last actual price and last predicted price
    last_actual_price = actual_prices[-1][0]
    last_predicted_price = future_predictions[-1][0]
    plt.annotate(f'Last Actual: {last_actual_price:.2f}', xy=(stock_data.index[-1], last_actual_price), xytext=(10, 10),
                 textcoords='offset points', ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.annotate(f'Last Predicted: {last_predicted_price:.2f}', xy=(future_dates[-1], last_predicted_price),
                 xytext=(10, -10),
                 textcoords='offset points', ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.savefig(os.path.join(plot_dir, f'{stock_symbol}_future_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Future prediction plot saved for {stock_symbol}")

    # Print paths to saved plots
    print(f"Plots saved in directory: {os.path.abspath(plot_dir)}")
    print(f"Historical and predicted prices plot: {os.path.join(plot_dir, f'{stock_symbol}_prediction.png')}")
    print(f"Future prediction plot: {os.path.join(plot_dir, f'{stock_symbol}_future_prediction.png')}")

    # Print some statistics
    print(f"\nStatistics for {stock_symbol}:")
    print(f"Data range: {stock_data.index[0]} to {stock_data.index[-1]}")
    print(f"Number of data points: {len(stock_data)}")
    print(f"Minimum price: {np.min(actual_prices):.2f}")
    print(f"Maximum price: {np.max(actual_prices):.2f}")
    print(f"Last actual price: {last_actual_price:.2f}")
    print(f"Last predicted price: {last_predicted_price:.2f}")


if __name__ == "__main__":
    stock_market_analysis('NVDA', test_ratio=0.2, future_days=30)