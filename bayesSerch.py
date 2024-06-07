#Update: 06/07/24, Code compliles but the Baysian optimization seems
#to be ineffective in improving predictions

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from scikeras.wrappers import KerasRegressor
import os


def create_model(lstm_units=50, dropout_rate=0.2, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True,
                   input_shape=(60, 11)))  # Adjusted input shape to match number of features
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units))
    model.add(Dense(50))
    model.add(Dense(11))  # Output shape: (num_features)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
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
    data['RSI'] = RSIIndicator(data['Close'], window=14).rsi().fillna(0)
    macd = MACD(data['Close'])
    data['MACD'] = macd.macd().fillna(0)
    data['MACD_Signal'] = macd.macd_signal().fillna(0)
    bollinger = BollingerBands(data['Close'])
    data['BB_High'] = bollinger.bollinger_hband().fillna(0)
    data['BB_Low'] = bollinger.bollinger_lband().fillna(0)
    data['50_MA'] = SMAIndicator(data['Close'], window=50).sma_indicator().fillna(0)
    data['100_MA'] = SMAIndicator(data['Close'], window=100).sma_indicator().fillna(0)

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
    features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'Rate of Return', '50_MA',
                '100_MA', 'Benchmark_Close']
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, shuffle=False)

    # Step 5: Define the Hyperparameter Space
    search_space = {
        'lstm_units': Integer(50, 200),
        'dropout_rate': Real(0.1, 0.5),
        'optimizer': ['adam', 'rmsprop']
    }

    # Step 6: Bayesian Optimization for Hyperparameter Tuning
    def print_status(result):
        iteration = result.func_vals.shape[0]  # Get the current iteration number
        params = result.x  # Get the current parameters being evaluated
        best_score = result.fun  # Get the best score found so far
        best_params = result.x_iters[np.argmin(result.func_vals)]  # Get the parameters corresponding to the best score
        print(f"Iteration {iteration}:")
        print(f"\t- Parameters: {params}")
        print(f"\t- Best score: {best_score}")
        print(f"\t- Best parameters: {best_params}")

    model = KerasRegressor(model=create_model, lstm_units=50, dropout_rate=0.29206627614880376, optimizer='adam',
                           epochs=10, batch_size=32, verbose=0)
    bayes_search = BayesSearchCV(estimator=model, search_spaces=search_space, n_iter=10, cv=3, verbose=1, n_jobs=-1)
    bayes_search.fit(x_train, y_train, callback=print_status)

    best_params = bayes_search.best_params_
    print(f"Best Parameters: {best_params}")

    # Step 7: Train the final model with best parameters
    final_model = create_model(lstm_units=best_params['lstm_units'], dropout_rate=best_params['dropout_rate'],
                               optimizer=best_params['optimizer'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
    model_dir = f"{stock_ticker}_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    checkpoint_path = os.path.join(model_dir, "best_model.keras")
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    final_model.fit(x_train, y_train, batch_size=32, epochs=30, validation_split=0.2,
                    callbacks=[early_stopping, model_checkpoint])

    # Save the final model architecture
    model_json = final_model.to_json()
    with open(os.path.join(model_dir, "final_model.json"), "w") as json_file:
        json_file.write(model_json)

    # Step 8: Make Predictions and Plot Results
    predictions = final_model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Align predictions with actual data index
    prediction_start_index = len(data) - len(predictions)
    prediction_dates = data.index[prediction_start_index:]

    valid = pd.DataFrame(index=prediction_dates, data=predictions,
                         columns=[f'Predictions_{feature}' for feature in features])

    plt.figure(figsize=(16, 28))

    # Stock Price Plot
    plt.subplot(6, 1, 1)
    plt.plot(data['Close'][:len(y_train) + time_step], label='Actual Prices (Train)')
    plt.plot(data['Close'][len(y_train) + time_step:], label='Actual Prices (Test)')
    plt.plot(valid.index, valid['Predictions_Close'], label='Predictions (Close)', color='green')
    plt.title(f'Stock Price Prediction of {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Benchmark Price Plot
    plt.subplot(6, 1, 2)
    plt.plot(data['Benchmark_Close'][:len(y_train) + time_step], label='Actual Prices (Train)')
    plt.plot(data['Benchmark_Close'][len(y_train) + time_step:], label='Actual Prices (Test)')
    plt.plot(valid.index, valid['Predictions_Close'], label='Predictions (Close)', color='green')
    plt.title(f'Stock Price Prediction of {benchmark_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    plt.tight_layout(pad=3.0)  # Adjust padding
    plt.subplots_adjust(hspace=0.5)  # Adjust space between plots
    plt.show()


# Example usage:
stock_market_analysis('NVDA', test_ratio=0.05)
