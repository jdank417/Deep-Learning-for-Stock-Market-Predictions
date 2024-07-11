import os
import pickle
import logging
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Input, Flatten, concatenate, Lambda, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import mse
import tensorflow.keras.backend as K
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

def evaluate_model(y_true, y_pred):
    mse_value = mean_squared_error(y_true, y_pred)
    mae_value = mean_absolute_error(y_true, y_pred)
    return mse_value, mae_value

def save_model_weights(model, model_name):
    model.save_weights(f"{model_name}_weights.h5")

def save_scaler(scaler, scaler_name):
    with open(f"{scaler_name}.pkl", "wb") as f:
        pickle.dump(scaler, f)

def load_model_weights(model, model_name):
    try:
        model.load_weights(f"{model_name}_weights.h5")
        print(f"{model_name} weights loaded successfully.")
    except Exception as e:
        print(f"Error loading {model_name} weights: {e}")

def load_scaler(scaler_name):
    try:
        with open(f"{scaler_name}.pkl", "rb") as f:
            scaler = pickle.load(f)
        print(f"{scaler_name} loaded successfully.")
        return scaler
    except Exception as e:
        print(f"Error loading {scaler_name}: {e}")
        return None

def log_metrics(symbol, mse_value, mae_value):
    logger.info(f"Evaluation Metrics for {symbol}:")
    logger.info(f"Mean Squared Error: {mse_value}")
    logger.info(f"Mean Absolute Error: {mae_value}")

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

    x = concatenate([Flatten()(inputs), Flatten()(condition)])
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    latent_inputs = concatenate([z, Flatten()(condition)])
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
    vae = create_conditional_vae(input_shape=input_shape, latent_dim=latent_dim, condition_shape=(input_shape[0] * input_shape[1],))
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    xgb = XGBRegressor(n_estimators=100, n_jobs=-1)

    def ensemble_predict(X):
        X = np.array(X)
        if X.ndim == 2:
            X = X.reshape(1, *X.shape)

        lstm_pred = lstm_cnn.predict(X)
        lstm_pred_flatten = lstm_pred.flatten().reshape(1, -1)

        # Ensure lstm_pred_flatten matches the expected condition shape
        expected_condition_shape = input_shape[0] * input_shape[1]
        if lstm_pred_flatten.shape[1] != expected_condition_shape:
            lstm_pred_flatten = np.resize(lstm_pred_flatten, (1, expected_condition_shape))

        vae_pred = vae.predict([X, lstm_pred_flatten])

        X_reshaped = X.reshape(X.shape[0], -1)
        rf_pred = rf.predict(X_reshaped)
        xgb_pred = xgb.predict(X_reshaped)

        # Ensure all predictions are reshaped correctly for averaging
        lstm_pred_resized = np.resize(lstm_pred.flatten(), expected_condition_shape)
        vae_pred_resized = np.resize(vae_pred.flatten(), expected_condition_shape)
        rf_pred_resized = np.resize(rf_pred.flatten(), expected_condition_shape)
        xgb_pred_resized = np.resize(xgb_pred.flatten(), expected_condition_shape)

        final_pred = (lstm_pred_resized + vae_pred_resized + rf_pred_resized + xgb_pred_resized) / 4
        return final_pred.reshape(1, -1)  # Ensure 2D output

    return ensemble_predict, lstm_cnn, vae, rf, xgb


def plot_predictions(data, symbol, window, scaler):
    data['Predicted'] = np.nan
    data['Predicted'][-len(window):] = window

    # Select only the columns used for scaling
    scaled_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'ATR', 'MA20', 'MA50', 'BB_high',
                      'BB_low', 'Stoch_k', 'Stoch_d', 'OBV']
    data_scaled = scaler.inverse_transform(data[scaled_columns])

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data_scaled[:, scaled_columns.index('Close')], label='Actual Close Prices', color='b')
    plt.plot(data.index, data['Predicted'], label='Predicted Close Prices', color='r')
    plt.title(f"{symbol} Stock Price Prediction")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def train_and_predict(symbol, start_date, end_date, prediction_days):
    data = yf.download(symbol, start=start_date, end=end_date)
    data = add_advanced_features(data)

    # Select only the columns to be scaled
    scaled_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'ATR', 'MA20', 'MA50', 'BB_high',
                      'BB_low', 'Stoch_k', 'Stoch_d', 'OBV']
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data[scaled_columns])
    save_scaler(scaler, "data_scaler")

    time_steps = 60
    X = []
    y = []

    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i])
        y.append(scaled_data[i, scaled_columns.index('Close')])

    X, y = np.array(X), np.array(y)
    input_shape = X.shape[1:]

    ensemble_predict, lstm_cnn, vae, rf, xgb = create_ensemble_model(input_shape)

    lstm_cnn.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    lstm_cnn.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    rf.fit(X.reshape(X.shape[0], -1), y)
    xgb.fit(X.reshape(X.shape[0], -1), y)

    predictions = []
    last_sequence = X[-1]
    for i in range(prediction_days):
        X_last = last_sequence.reshape(1, *X.shape[1:])
        pred = ensemble_predict(X_last)
        predictions.append(pred[0, scaled_columns.index('Close')])  # Store only the close price prediction

        # Reshape pred to match the dimensions of last_sequence[1:]
        pred_reshaped = pred.reshape(-1, X.shape[2])
        new_data = np.concatenate((last_sequence[1:], pred_reshaped), axis=0)
        last_sequence = new_data[-time_steps:]  # Keep only the last time_steps rows

    # Inverse transform the data and predictions
    scaled_data = scaler.inverse_transform(scaled_data)
    data[scaled_columns] = scaled_data

    # Inverse transform the predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(
        np.hstack([np.zeros((len(predictions), len(scaled_columns) - 1)), predictions]))[:, -1]

    prediction_series = np.zeros(len(data) + prediction_days)
    prediction_series[:len(data)] = data['Close'].values
    prediction_series[-prediction_days:] = predictions

    data['Predicted_Close'] = prediction_series[:len(data)]

    mse_value, mae_value = evaluate_model(data['Close'].values[-prediction_days:], prediction_series[-prediction_days:])
    log_metrics(symbol, mse_value, mae_value)

    plot_predictions(data, symbol, prediction_series[-prediction_days:], scaler)


# Example usage
train_and_predict('AAPL', '2015-01-01', '2022-12-31', prediction_days=30)
