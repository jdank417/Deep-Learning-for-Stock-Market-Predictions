import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import os

def stock_market_analysis(stock_ticker, start_date="2010-01-01", end_date="2024-01-01", test_ratio=0.2, benchmark_ticker='VOO'):
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

    # Step 4: Prepare Data for LSTM
    features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'Rate of Return', '50_MA', 'Benchmark_Close']
    stock_data = data[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)

    def create_dataset(data, time_step=1):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), :])
            Y.append(data[i + time_step, :])  # Target is all features
        return np.array(X), np.array(Y)

    time_step = 60  # Re-adjusting time step to a higher value
    x, y = create_dataset(scaled_data, time_step)
    train_size = int(len(x) * (1 - test_ratio))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Step 5: Build the LSTM Model with Dropout Regularization
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(time_step, len(features))))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(50))
    model.add(Dense(len(features)))  # Output shape: (num_features)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Step 6: Train the model in stages and plot after each stage
    epochs_per_stage = 10
    num_stages = 3

    # Directory to save model and results
    model_dir = f"{stock_ticker}_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

    for stage in range(num_stages):
        print(f"Training stage {stage + 1}/{num_stages}...")
        # Model Checkpoint to save the best model
        checkpoint_path = os.path.join(model_dir, f"stage_{stage}_best_model.keras")
        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        model.fit(x_train, y_train, batch_size=32, epochs=epochs_per_stage, validation_split=0.2,
                  callbacks=[early_stopping, model_checkpoint])

        # Save the model architecture
        model_json = model.to_json()
        with open(os.path.join(model_dir, f"stage_{stage}_model.json"), "w") as json_file:
            json_file.write(model_json)

        # Make Predictions after each stage
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Align predictions with actual data index
        prediction_start_index = len(data) - len(predictions)
        prediction_dates = data.index[prediction_start_index:]

        valid = pd.DataFrame(index=prediction_dates, data=predictions,
                             columns=[f'Predictions_{feature}' for feature in features])

        plt.figure(figsize=(16, 28))  # Increased figure size

        # Stock Price Plot
        plt.subplot(6, 1, 1)
        plt.plot(data['Close'][:train_size + time_step], label='Actual Prices (Train)')
        plt.plot(data['Close'][train_size + time_step:], label='Actual Prices (Test)')
        plt.plot(valid.index, valid['Predictions_Close'], label='Predictions (Close)', color='green')
        plt.title(f'Stock Price Prediction of {stock_ticker}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # Benchmark Price Plot
        plt.subplot(6, 1, 2)
        plt.plot(data['Benchmark_Close'][:train_size + time_step], label='Actual Prices (Train)')
        plt.plot(data['Benchmark_Close'][train_size + time_step:], label='Actual Prices (Test)')
        plt.plot(valid.index, valid['Predictions_Close'], label='Predictions (Close)', color='green')
        plt.title(f'Stock Price Prediction of {benchmark_ticker}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # Trading Volume Plot
        plt.subplot(6, 1, 3)
        plt.plot(data['Volume'][:train_size + time_step], label='Actual Volume (Train)')
        plt.plot(data['Volume'][train_size + time_step:], label='Actual Volume (Test)')
        plt.plot(valid.index, valid['Predictions_Volume'], label='Predictions (Volume)', color='green')
        plt.title(f'Trading Volume Prediction of {stock_ticker}')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.legend()

        # RSI Plot
        plt.subplot(6, 1, 4)
        plt.plot(data['RSI'][:train_size + time_step], label='Actual RSI (Train)')
        plt.plot(data['RSI'][train_size + time_step:], label='Actual RSI (Test)')
        plt.plot(valid.index, valid['Predictions_RSI'], label='Predictions (RSI)', color='green')
        plt.title(f'RSI Prediction of {stock_ticker}')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()

        # MACD Plot
        plt.subplot(6, 1, 5)
        plt.plot(data['MACD'][:train_size + time_step], label='Actual MACD (Train)')
        plt.plot(data['MACD'][train_size + time_step:], label='Actual MACD (Test)')
        plt.plot(valid.index, valid['Predictions_MACD'], label='Predictions (MACD)', color='green')
        plt.title(f'MACD Prediction of {stock_ticker}')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()

        # Moving Average Plot
        plt.subplot(6, 1, 6)
        plt.plot(data['50_MA'][:train_size + time_step], label='Actual 50-day MA (Train)')
        plt.plot(data['50_MA'][train_size + time_step:], label='Actual 50-day MA (Test)')
        plt.plot(valid.index, valid['Predictions_50_MA'], label='Predictions (50-day MA)', color='green')
        plt.title(f'50-day Moving Average Prediction of {stock_ticker}')
        plt.xlabel('Date')
        plt.ylabel('50-day MA')
        plt.legend()

        plt.tight_layout(pad=3.0)  # Adjust padding
        plt.subplots_adjust(hspace=0.5)  # Adjust space between plots
        plt.show()


# Example usage:
stock_market_analysis('NVDA', test_ratio=0.05)