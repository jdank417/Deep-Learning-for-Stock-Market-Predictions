import os
import logging
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, LSTM, Dropout, Conv1D, MaxPooling1D,
                                     Input, Reshape, Lambda, Flatten, concatenate, Attention)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import mse
import tensorflow.keras.backend as K
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pywt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Create a 'logs' directory if it doesn't exist
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, 'stock_prediction.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stock_prediction')


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal


def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def create_lstm_cnn_attention_encoder(lstm_units=100, conv_filters=64, conv_kernel_size=3, dropout_rate=0.3,
                                      time_steps=60, num_features=4):
    inputs = Input(shape=(time_steps, num_features))
    x = Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    lstm_out = LSTM(units=lstm_units, return_sequences=True)(x)
    attention = Attention()([lstm_out, lstm_out])
    x = concatenate([lstm_out, attention])
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    features = Dense(50, activation='relu', name='features')(x)
    outputs = Dense(num_features)(features)

    encoder = Model(inputs, [outputs, features], name='lstm_cnn_attention_encoder')
    encoder.compile(optimizer='adam', loss='mean_squared_error')
    return encoder


def create_conditional_vae(input_shape, latent_dim, condition_shape):
    inputs = Input(shape=input_shape, name='vae_input')
    condition = Input(shape=condition_shape, name='condition_input')

    x = concatenate([Flatten()(inputs), condition])
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    latent_inputs = concatenate([z, condition])
    x = Dense(64, activation='relu')(latent_inputs)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(np.prod(input_shape), activation='linear')(x)
    outputs = Reshape(input_shape)(outputs)

    vae = Model([inputs, condition], outputs)

    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return vae


def add_advanced_features(data):
    data = data.copy()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'], _ = calculate_macd(data['Close'])
    data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'])
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)
    return data


def wavelet_features(data, wavelet='db1', level=3):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return np.concatenate(coeffs)


def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, :])
        y.append(data[i, 0])  # Predict 'Close' price
    return np.array(X), np.array(y)


def create_ensemble_model(input_shape, latent_dim=8):
    lstm_cnn = create_lstm_cnn_attention_encoder(time_steps=input_shape[0], num_features=input_shape[1])
    vae = create_conditional_vae(input_shape=input_shape, latent_dim=latent_dim, condition_shape=(50,))
    rf = RandomForestRegressor(n_estimators=100)
    xgb = XGBRegressor(n_estimators=100)
    gp = GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)), n_restarts_optimizer=10, alpha=0.1)

    def ensemble_predict(X):
        lstm_pred, lstm_features = lstm_cnn.predict(X)
        vae_pred = vae.predict([X, lstm_features])
        rf_pred = rf.predict(X.reshape(X.shape[0], -1))
        xgb_pred = xgb.predict(X.reshape(X.shape[0], -1))
        gp_pred, _ = gp.predict(X.reshape(X.shape[0], -1), return_std=True)

        # Ensure all predictions have the same shape
        if len(lstm_pred.shape) == 3:
            lstm_pred = lstm_pred[:, -1, :]  # Use only the last time step
        if len(vae_pred.shape) == 3:
            vae_pred = vae_pred[:, -1, :]  # Use only the last time step
        rf_pred = rf_pred.reshape(-1, 1)
        xgb_pred = xgb_pred.reshape(-1, 1)
        gp_pred = gp_pred.reshape(-1, 1)

        # Combine predictions for all features
        combined_pred = np.column_stack((lstm_pred, vae_pred, rf_pred, xgb_pred, gp_pred))

        return np.mean(combined_pred, axis=1)

    return ensemble_predict, (lstm_cnn, vae, rf, xgb, gp)

def sliding_window_train(model, X, y, window_size=1000):
    for i in range(0, len(X) - window_size, 100):  # Step by 100 for efficiency
        X_window = X[i:i + window_size]
        y_window = y[i:i + window_size]
        model.fit(X_window, y_window, epochs=1, verbose=0)
    return model


def backtest(model, X, y, initial_window=1000):
    predictions = []
    for i in range(initial_window, len(X)):
        model.fit(X[:i], y[:i])
        pred = model.predict(X[i].reshape(1, -1))
        predictions.append(pred[0])
    return np.array(predictions)


def generate_future_predictions_hybrid(ensemble_model, last_sequence, future_days, scaler, num_features):
    future_predictions = []
    current_sequence = last_sequence.reshape(1, *last_sequence.shape)

    for _ in range(future_days):
        next_prediction = ensemble_model(current_sequence)
        future_predictions.append(next_prediction[0])

        # Update sequence for next iteration
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1] = next_prediction  # Update all features

    future_predictions = np.array(future_predictions).reshape(-1, 1)

    print(f"Future Predictions Shape: {future_predictions.shape}")
    print(f"Scaler Min Shape: {scaler.min_.shape}, Scaler Scale Shape: {scaler.scale_.shape}")

    # Prepare the predictions for inverse transform
    future_predictions_padded = np.zeros((future_predictions.shape[0], scaler.scale_.shape[0]))
    future_predictions_padded[:, 0] = future_predictions.flatten()

    # Inverse transform predictions
    future_predictions_rescaled = scaler.inverse_transform(future_predictions_padded)

    return future_predictions_rescaled[:, 0]  # Return only the first column (closing price)


def plot_results(stock_data, predictions_full, future_predictions_df, train_size, time_steps):
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data['Close'], label='Actual Stock Price')
    plt.plot(stock_data.index[train_size + time_steps:], predictions_full[train_size + time_steps:], label='Smoothed Predictions')
    plt.plot(future_predictions_df.index, future_predictions_df['Future Predictions'], label='Future Predictions')
    plt.axvline(x=stock_data.index[train_size + time_steps], color='r', linestyle='--', label='Train-Test Split')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()


def stock_market_analysis_with_hybrid_model(stock_symbol, test_ratio, future_days):
    logger.info(f"Starting analysis for stock symbol: {stock_symbol}")
    logger.info(f"Test ratio: {test_ratio}, Future days: {future_days}")

    # Download stock data
    logger.info(f"Downloading stock data for {stock_symbol}")
    stock_data = yf.download(stock_symbol, start='2022-01-01', end='2024-01-01')
    logger.info(f"Downloaded data shape: {stock_data.shape}")

    if stock_data.empty:
        logger.error("Failed to retrieve data for the given stock symbol")
        raise ValueError("Failed to retrieve data for the given stock symbol")

    # Add technical indicators
    logger.info("Adding technical indicators")
    stock_data = add_advanced_features(stock_data)
    logger.info(f"Technical indicators added. Updated data shape: {stock_data.shape}")

    # Scale the data
    logger.info("Scaling the data")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)
    logger.info("Data scaled successfully")

    # Scale the closing prices separately
    closing_price_scaler = MinMaxScaler(feature_range=(0, 1))
    closing_prices = stock_data['Close'].values.reshape(-1, 1)
    closing_prices_scaled = closing_price_scaler.fit_transform(closing_prices)

    # Add wavelet features
    wavelet_feats = np.apply_along_axis(wavelet_features, 0, scaled_data)

    # Ensure arrays have the same number of rows
    min_rows = min(scaled_data.shape[0], wavelet_feats.shape[0])
    scaled_data = scaled_data[:min_rows]
    wavelet_feats = wavelet_feats[:min_rows]

    # Combine scaled data and wavelet features
    combined_data = np.hstack((scaled_data, wavelet_feats))

    # Create a new scaler for the combined data
    combined_scaler = MinMaxScaler(feature_range=(0, 1))
    combined_scaled_data = combined_scaler.fit_transform(combined_data)

    # Update num_features
    num_features = combined_scaled_data.shape[1]

    # Prepare the dataset
    logger.info("Preparing dataset")
    time_steps = 60
    X, y = create_dataset(combined_scaled_data, time_steps)
    logger.info(f"Dataset prepared. X shape: {X.shape}, y shape: {y.shape}")

    # Split the data
    train_size = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create ensemble model
    ensemble_model, (lstm_cnn, vae, rf, xgb, gp) = create_ensemble_model((time_steps, num_features))

    # Train models
    lstm_cnn.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    vae.fit([X_train, lstm_cnn.predict(X_train)[1]], X_train, epochs=50, batch_size=32, validation_split=0.2)
    rf.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    xgb.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    gp.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    # Continuous updating with sliding window
    lstm_cnn = sliding_window_train(lstm_cnn, X_train, y_train)

    # Backtesting
    backtest_predictions = backtest(lstm_cnn, X, y)

    # Generate future predictions
    future_predictions = generate_future_predictions_hybrid(ensemble_model, X_train[-1], future_days, combined_scaler, num_features)

    # Debug print for future predictions
    print("Future Predictions:", future_predictions)

    # Prepare full dataset predictions for plotting
    test_predictions = ensemble_model(X_test)
    test_predictions = test_predictions.reshape(-1, 1)  # Reshape to 2D array for inverse_transform

    # Debug print for test predictions before inverse transform
    print("Raw Test Predictions:", test_predictions)

    # Inverse transform the closing price predictions using the separate scaler
    test_predictions_rescaled = closing_price_scaler.inverse_transform(test_predictions)

    # Debug print for test predictions after inverse transform
    print("Rescaled Test Predictions:", test_predictions_rescaled)

    # Prepare predictions for plotting
    predictions_full = np.zeros((len(stock_data), 1))
    predictions_full[:] = np.nan
    predictions_full[train_size + time_steps:] = test_predictions_rescaled.reshape(-1, 1)

    # Extend the stock_data index for future dates
    last_date = stock_data.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

    # Create a DataFrame for the future predictions
    future_predictions_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Future Predictions'])

    # Plot the results
    plot_results(stock_data, predictions_full, future_predictions_df, train_size, time_steps)

    logger.info("Analysis completed successfully")





if __name__ == "__main__":
    stock_market_analysis_with_hybrid_model('NVDA', test_ratio=0.2, future_days=90)