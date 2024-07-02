import os
import logging
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Input, Reshape, Lambda, Flatten, \
    concatenate, Attention
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import mse
import tensorflow.keras.backend as K
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pywt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator
from joblib import Parallel, delayed
import tensorflow as tf

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'stock_prediction.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stock_prediction')


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
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
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def add_advanced_features(data):
    data = data.copy()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'], _ = calculate_macd(data['Close'])
    data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'])
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # Adding Bollinger Bands
    bb_indicator = BollingerBands(close=data["Close"], window=20, window_dev=2)
    data['BB_high'] = bb_indicator.bollinger_hband()
    data['BB_low'] = bb_indicator.bollinger_lband()

    # Adding Stochastic Oscillator
    stoch = StochasticOscillator(high=data["High"], low=data["Low"], close=data["Close"], window=14, smooth_window=3)
    data['Stoch_k'] = stoch.stoch()
    data['Stoch_d'] = stoch.stoch_signal()

    # Adding On-Balance Volume
    obv = OnBalanceVolumeIndicator(close=data["Close"], volume=data["Volume"])
    data['OBV'] = obv.on_balance_volume()

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


def create_ensemble_model(input_shape, latent_dim=8):
    lstm_cnn = create_lstm_cnn_attention_encoder(time_steps=input_shape[0], num_features=input_shape[1])
    vae = create_conditional_vae(input_shape=input_shape, latent_dim=latent_dim, condition_shape=(50,))
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    xgb = XGBRegressor(n_estimators=100, n_jobs=-1)
    gp = GaussianProcessRegressor(kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)), n_restarts_optimizer=10, alpha=0.1)

    def ensemble_predict(X):
        # Convert X to numpy array if it's a list
        X = np.array(X)

        # Ensure X has the correct shape
        if X.ndim == 2:
            X = X.reshape(1, *X.shape)

        lstm_pred, lstm_features = lstm_cnn.predict(X)
        vae_pred = vae.predict([X, lstm_features])
        X_reshaped = X.reshape(X.shape[0], -1)
        rf_pred = rf.predict(X_reshaped)
        xgb_pred = xgb.predict(X_reshaped)
        gp_pred, _ = gp.predict(X_reshaped, return_std=True)

        # Ensure all predictions have the same shape
        lstm_pred = lstm_pred.flatten()
        vae_pred = vae_pred.flatten()
        rf_pred = rf_pred.flatten()
        xgb_pred = xgb_pred.flatten()
        gp_pred = gp_pred.flatten()

        # Ensure all predictions have the same length
        min_length = min(len(lstm_pred), len(vae_pred), len(rf_pred), len(xgb_pred), len(gp_pred))
        lstm_pred = lstm_pred[:min_length]
        vae_pred = vae_pred[:min_length]
        rf_pred = rf_pred[:min_length]
        xgb_pred = xgb_pred[:min_length]
        gp_pred = gp_pred[:min_length]

        combined_predictions = np.mean(
            [lstm_pred, vae_pred, rf_pred, xgb_pred, gp_pred],
            axis=0
        )
        return combined_predictions

    return ensemble_predict, (lstm_cnn, vae, rf, xgb, gp)


def generate_future_predictions(ensemble_model, last_input, future_days, scaler, num_features):
    future_predictions = []
    input_sequence = last_input.copy()

    for _ in range(future_days):
        input_sequence = np.roll(input_sequence, -1, axis=0)
        prediction = ensemble_model(input_sequence.reshape(1, input_sequence.shape[0], num_features))
        input_sequence[-1, 0] = prediction[0]  # Use only the first prediction
        future_predictions.append(prediction[0])

    # Create a dummy array with the same number of features as the original data
    dummy_array = np.zeros((len(future_predictions), num_features))
    dummy_array[:, 0] = future_predictions  # Set the first column (closing price) to our predictions

    # Inverse transform the entire dummy array
    inverse_transformed = scaler.inverse_transform(dummy_array)

    # Return only the first column (closing price)
    return inverse_transformed[:, 0]


def stock_market_analysis(symbol, start_date, end_date, time_steps=60, future_days=90):
    # Download stock data
    df = yf.download(symbol, start=start_date, end=end_date)
    df = add_advanced_features(df)

    # Normalize features
    feature_cols = ['Close', 'RSI', 'MACD', 'ATR', 'MA20', 'MA50', 'BB_high', 'BB_low', 'Stoch_k', 'Stoch_d', 'OBV']
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    # Create datasets
    X, y = create_dataset(scaled_data, time_steps)
    split_ratio = 0.8
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Create ensemble model and fit
    ensemble_model, model_components = create_ensemble_model(X_train.shape[1:])
    lstm_cnn, vae, rf, xgb, gp = model_components

    # Train LSTM-CNN-Attention Encoder
    lstm_cnn.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    # Train Conditional VAE
    vae.fit([X_train, lstm_cnn.predict(X_train)[1]], epochs=50, batch_size=64, validation_split=0.2, verbose=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    # Train other models
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    rf.fit(X_train_reshaped, y_train)
    xgb.fit(X_train_reshaped, y_train)
    gp.fit(X_train_reshaped, y_train)

    # Predict and plot results
    y_pred = ensemble_model(X_test)

    # Inverse transform y_test and y_pred
    y_test_inv = scaler.inverse_transform(np.column_stack([y_test, np.zeros((len(y_test), len(feature_cols) - 1))]))[:,
                 0]
    y_pred_inv = scaler.inverse_transform(np.column_stack([y_pred, np.zeros((len(y_pred), len(feature_cols) - 1))]))[:,
                 0]

    # Generate future predictions
    future_predictions = generate_future_predictions(ensemble_model, X_test[-1], future_days, scaler, len(feature_cols))

    # Create date ranges
    historical_dates = df.index[-len(y_test):]
    future_dates = pd.date_range(start=historical_dates[-1] + pd.Timedelta(days=1), periods=future_days, freq='D')

    plt.figure(figsize=(16, 8))
    plt.plot(historical_dates, y_test_inv, label='Actual Prices', color='blue')
    plt.plot(historical_dates, y_pred_inv, label='Predicted Prices', color='red')
    plt.plot(future_dates, future_predictions, label='Future Predictions', color='green')

    plt.title(f'{symbol} Stock Price Prediction', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Improve x-axis date formatting
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    # Optionally, use a logarithmic scale if the price range is very wide
    # plt.yscale('log')

    plt.show()

    # Optional: Print some performance metrics
    mse = np.mean((y_test_inv - y_pred_inv) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_inv - y_pred_inv))

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")


if __name__ == '__main__':
    stock_market_analysis('NVDA', '2020-01-01', '2024-01-01', time_steps=60, future_days=90)