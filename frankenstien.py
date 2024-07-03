import os
import logging
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Input, Flatten, concatenate, Lambda, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import mse
import tensorflow.keras.backend as K
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator

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

def interpolate_data(data):
    return data.interpolate(method='linear', axis=0).ffill().bfill()

def add_advanced_features(data):
    data = data.copy()
    data = interpolate_data(data)  # Interpolate missing data
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'], _ = calculate_macd(data['Close'])
    data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'])
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    bb_indicator = BollingerBands(close=data["Close"], window=20, window_dev=2)
    data['BB_high'] = bb_indicator.bollinger_hband()
    data['BB_low'] = bb_indicator.bollinger_lband()

    stoch = StochasticOscillator(high=data["High"], low=data["Low"], close=data["Close"], window=14, smooth_window=3)
    data['Stoch_k'] = stoch.stoch()
    data['Stoch_d'] = stoch.stoch_signal()

    obv = OnBalanceVolumeIndicator(close=data["Close"], volume=data["Volume"])
    data['OBV'] = obv.on_balance_volume()

    data.dropna(inplace=True)
    return data

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

def create_lstm_cnn_attention_encoder(lstm_units=100, conv_filters=64, conv_kernel_size=3, dropout_rate=0.3, time_steps=60, num_features=11):
    inputs = Input(shape=(time_steps, num_features))
    x = Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(units=lstm_units, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units=lstm_units)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_features, name='output')(x)

    encoder = Model(inputs, outputs, name='lstm_cnn_attention_encoder')
    encoder.compile(optimizer='adam', loss='mean_squared_error')
    return encoder

def create_ensemble_model(input_shape, latent_dim=8):
    lstm_cnn = create_lstm_cnn_attention_encoder(time_steps=input_shape[0], num_features=input_shape[1])
    vae = create_conditional_vae(input_shape=input_shape, latent_dim=latent_dim, condition_shape=(input_shape[1],))
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    xgb = XGBRegressor(n_estimators=100, n_jobs=-1)

    def ensemble_predict(X):
        X = np.array(X)
        if X.ndim == 2:
            X = X.reshape(1, *X.shape)

        lstm_pred = lstm_cnn.predict(X)
        vae_pred = vae.predict([X, lstm_pred])
        X_reshaped = X.reshape(X.shape[0], -1)
        rf_pred = rf.predict(X_reshaped)
        xgb_pred = xgb.predict(X_reshaped)

        lstm_pred = lstm_pred.flatten()
        vae_pred = vae_pred.flatten()
        rf_pred = rf_pred.flatten()
        xgb_pred = xgb_pred.flatten()

        min_length = min(len(lstm_pred), len(vae_pred), len(rf_pred), len(xgb_pred))
        lstm_pred = lstm_pred[:min_length]
        vae_pred = vae_pred[:min_length]
        rf_pred = rf_pred[:min_length]
        xgb_pred = xgb_pred[:min_length]

        combined_predictions = np.mean([lstm_pred, vae_pred, rf_pred, xgb_pred], axis=0)
        return combined_predictions

    return ensemble_predict, (lstm_cnn, vae, rf, xgb)

def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, :])
        y.append(data[i])  # Predict the entire feature set
    return np.array(X), np.array(y)

def generate_future_predictions(ensemble_model, last_input, future_days, scaler, num_features):
    future_predictions = []
    input_sequence = last_input.copy()

    for _ in range(future_days):
        input_sequence = np.roll(input_sequence, -1, axis=0)
        prediction = ensemble_model(input_sequence.reshape(1, input_sequence.shape[0], num_features))
        input_sequence[-1] = prediction  # Use the entire prediction set
        future_predictions.append(prediction)

    future_predictions = np.array(future_predictions).reshape(-1, num_features)
    inverse_transformed = scaler.inverse_transform(future_predictions)
    return inverse_transformed[:, 0]

def stock_market_analysis(symbol, start_date, end_date, time_steps=60, future_days=90):
    df = yf.download(symbol, start=start_date, end=end_date)
    df = add_advanced_features(df)

    feature_columns = ['Close', 'RSI', 'MACD', 'ATR', 'MA20', 'MA50', 'BB_high', 'BB_low', 'Stoch_k', 'Stoch_d', 'OBV']
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])

    X, y = create_dataset(scaled_data, time_steps)
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    ensemble_predict, models = create_ensemble_model(input_shape=(time_steps, X.shape[2]))

    lstm_cnn, vae, rf, xgb = models
    lstm_cnn.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    features = lstm_cnn.predict(X_train)
    vae.fit([X_train, features], epochs=50, batch_size=64, validation_split=0.2, verbose=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

    rf.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    xgb.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    last_input = X_test[-1]
    future_predictions = generate_future_predictions(ensemble_predict, last_input, future_days, scaler, X.shape[2])

    plt.figure(figsize=(14, 7))
    plt.plot(df.index[-len(y_test):], scaler.inverse_transform(scaled_data[-len(y_test):])[:, 0], color='blue', label='Actual Price')
    plt.plot(pd.date_range(start=df.index[-1], periods=future_days, freq='B'), future_predictions, color='red', label='Future Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.legend()
    plt.show()

# Example usage
if __name__ == '__main__':
    stock_market_analysis('NVDA', '2020-01-01', '2024-01-01', time_steps=60, future_days=90)