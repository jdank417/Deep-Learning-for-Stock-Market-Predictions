# This script provides a comprehensive pipeline for stock price prediction,
# integrating CNN for feature extraction and LSTM for sequence modeling,
# demonstrating a hybrid approach to capture both spatial and temporal patterns in stock data.
# Adjustments can be made to hyperparameters, model architecture, or data preprocessing steps
# based on specific requirements or performance evaluations.


import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import logging

# Create a 'logs' directory if it doesn't exist
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(filename=os.path.join(log_dir, 'stock_prediction.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stock_prediction')

def create_model(lstm_units=100, conv_filters=64, conv_kernel_size=3, dropout_rate=0.3, time_steps=60, num_features=4):
    model = Sequential()
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu',
                     input_shape=(time_steps, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def add_technical_indicators(data):
    data = data.copy()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)
    return data

def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, :])
        y.append(data[i, 0])  # Predict 'Close' price
    return np.array(X), np.array(y)

def stock_market_analysis_with_cnn_lstm(stock_symbol, test_ratio, future_days):
    logger.info(f"Starting analysis for stock symbol: {stock_symbol}")
    logger.info(f"Test ratio: {test_ratio}, Future days: {future_days}")

    # Download stock data
    logger.info(f"Downloading stock data for {stock_symbol}")
    stock_data = yf.download(stock_symbol, start='2010-01-01', end='2024-01-01')
    logger.info(f"Downloaded data shape: {stock_data.shape}")

    stock_data = stock_data[['Close', 'Volume']]

    if stock_data.empty:
        logger.error("Failed to retrieve data for the given stock symbol")
        raise ValueError("Failed to retrieve data for the given stock symbol")

    # Add technical indicators
    logger.info("Adding technical indicators")
    stock_data = add_technical_indicators(stock_data)
    logger.info(f"Technical indicators added. Updated data shape: {stock_data.shape}")

    # Scale the data
    logger.info("Scaling the data")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)
    logger.info("Data scaled successfully")

    # Prepare the dataset
    logger.info("Preparing dataset")
    time_steps = 60
    num_features = scaled_data.shape[1]
    X, y = create_dataset(scaled_data, time_steps)
    logger.info(f"Dataset prepared. X shape: {X.shape}, y shape: {y.shape}")

    # Split the data into train and validation sets for interpolation
    train_size = int(len(X) * 0.8)  # Use 80% for training
    X_train_interp, y_train_interp = X[:train_size], y[:train_size]
    X_val_interp, y_val_interp = X[train_size:], y[train_size:]

    # Define callbacks for interpolation
    model_checkpoint_callback_interp = ModelCheckpoint(
        filepath='Utils/interpolation_model_weights.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback_interp = EarlyStopping(monitor='val_loss', patience=5)

    # Pre-train the model on the interpolation task
    logger.info("Pre-training model on interpolation task")
    model = create_model(time_steps=time_steps, num_features=num_features)
    model.fit(X_train_interp, y_train_interp, validation_data=(X_val_interp, y_val_interp), epochs=50, batch_size=32,
              callbacks=[model_checkpoint_callback_interp, early_stopping_callback_interp])

    # Split the data into train and test sets for extrapolation
    train_size = int(len(X) * 0.9)  # Use 90% for training, leaving 10% for testing extrapolation
    X_train_extrap, y_train_extrap = X[:train_size], y[:train_size]
    X_test_extrap, y_test_extrap = X[train_size:], y[train_size:]

    # Load the pre-trained weights from the interpolation task
    logger.info("Loading pre-trained weights for extrapolation task")
    model = create_model(time_steps=time_steps, num_features=num_features)
    model.load_weights('interpolation_model_weights.h5')

    # Define callbacks for extrapolation
    model_checkpoint_callback_extrap = ModelCheckpoint(
        filepath='Utils/extrapolation_model_weights.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback_extrap = EarlyStopping(monitor='val_loss', patience=5)

    # Fine-tune the model on the extrapolation task
    logger.info("Fine-tuning model on extrapolation task")
    model.fit(X_train_extrap, y_train_extrap, validation_data=(X_test_extrap, y_test_extrap), epochs=20, batch_size=32,
              callbacks=[model_checkpoint_callback_extrap, early_stopping_callback_extrap])

    # Make predictions
    logger.info("Making predictions")
    predictions = model.predict(X_test_extrap)

    # Rescale predictions
    logger.info("Rescaling predictions")
    predictions = scaler.inverse_transform(
        np.concatenate((predictions, np.zeros((predictions.shape[0], num_features - 1))), axis=1))[:, 0]

    # Prepare full dataset predictions for plotting
    predictions_full = np.zeros((len(stock_data), 1))
    predictions_full[:] = np.nan
    predictions_full[train_size + time_steps:] = predictions.reshape(-1, 1)

    # Predict future stock prices
    last_sequence = scaled_data[-time_steps:]

    # Print initial last_sequence
    logger.info(f"Initial last_sequence shape: {last_sequence.shape}")
    logger.info(f"Initial last_sequence sample: {last_sequence[:5]}")

    future_predictions = []

    for _ in range(future_days):
        next_prediction = model.predict(last_sequence.reshape(1, time_steps, num_features))
        future_predictions.append(next_prediction[0, 0])
        next_prediction_full = np.concatenate((next_prediction, np.zeros((1, num_features - 1))), axis=1)
        last_sequence = np.concatenate((last_sequence[1:], next_prediction_full), axis=0)

        # Print updated last_sequence during each iteration
        logger.info(f"Updated last_sequence shape during iteration: {last_sequence.shape}")
        logger.info(f"Updated last_sequence sample during iteration: {last_sequence[-5:]}")

    # Convert future_predictions to a NumPy array
    future_predictions = np.array(future_predictions)

    # Print future_predictions before rescaling
    logger.info(f"Future predictions before rescaling: {future_predictions}")

    # Rescale future predictions
    future_predictions = scaler.inverse_transform(
        np.concatenate((future_predictions.reshape(-1, 1), np.zeros((future_predictions.shape[0], num_features - 1))),
                       axis=1))[:, 0]

    # Print future_predictions after rescaling
    logger.info(f"Future predictions after rescaling: {future_predictions}")

    # Extend the stock_data index for future dates
    last_date = stock_data.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

    # Create a DataFrame for the future predictions
    future_predictions_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Future Predictions'])

    # Print the future_predictions_df to debug
    logger.info("Future predictions DataFrame:")
    logger.info(future_predictions_df.head())

    # Plot the results
    logger.info("Plotting results")
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], color='blue', label='Actual Stock Price')
    plt.plot(pd.DataFrame(predictions_full, index=stock_data.index), color='orange', label='Predicted Stock Price')
    plt.plot(future_predictions_df.index, future_predictions_df['Future Predictions'], color='red', linestyle='--',
             label='Future Predictions')
    plt.fill_between(stock_data.index[train_size + time_steps:],
                     stock_data['Close'].values[train_size + time_steps:].flatten(),
                     predictions_full[train_size + time_steps:].flatten(), color='orange', alpha=0.3)
    plt.title(f'{stock_symbol} Stock Price Prediction with CNN-LSTM')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# Call the function with the stock symbol, desired test ratio, and future days to predict
stock_market_analysis_with_cnn_lstm('NVDA', test_ratio=0.2, future_days=30)
