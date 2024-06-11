import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


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
    data = data.copy()  # Make an explicit copy of the DataFrame slice
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Volume'] = data['Volume'].astype(float)  # Ensure volume is float
    data.dropna(inplace=True)
    return data


def stock_market_analysis_with_cnn_lstm(stock_symbol, test_ratio=0.2, future_days=30):
    # Download stock data
    stock_data = yf.download(stock_symbol, start='2010-01-01', end='2024-01-01')
    stock_data = stock_data[['Close', 'Volume']]

    if stock_data.empty:
        raise ValueError("Failed to retrieve data for the given stock symbol")

    # Add technical indicators
    stock_data = add_technical_indicators(stock_data)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)

    # Prepare the dataset
    time_steps = 60
    num_features = scaled_data.shape[1]

    def create_dataset(data):
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i - time_steps:i, :])
            y.append(data[i, 0])  # Predict 'Close' price
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data)

    # Split the data into train and validation sets for interpolation
    train_size = int(len(X) * 0.8)  # Use 80% for training
    X_train_interp, y_train_interp = X[:train_size], y[:train_size]
    X_val_interp, y_val_interp = X[train_size:], y[train_size:]

    # Define callbacks for interpolation
    model_checkpoint_callback_interp = ModelCheckpoint(
        filepath='interpolation_model_weights.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback_interp = EarlyStopping(monitor='val_loss', patience=5)

    # Pre-train the model on the interpolation task
    model = create_model(time_steps=time_steps, num_features=num_features)
    model.fit(X_train_interp, y_train_interp, validation_data=(X_val_interp, y_val_interp), epochs=50, batch_size=32,
              callbacks=[model_checkpoint_callback_interp, early_stopping_callback_interp])

    # Split the data into train and test sets for extrapolation
    train_size = int(len(X) * 0.9)  # Use 90% for training, leaving 10% for testing extrapolation
    X_train_extrap, y_train_extrap = X[:train_size], y[:train_size]
    X_test_extrap, y_test_extrap = X[train_size:], y[train_size:]

    # Load the pre-trained weights from the interpolation task
    model = create_model(time_steps=time_steps, num_features=num_features)
    model.load_weights('interpolation_model_weights.h5')

    # Define callbacks for extrapolation
    model_checkpoint_callback_extrap = ModelCheckpoint(
        filepath='extrapolation_model_weights.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback_extrap = EarlyStopping(monitor='val_loss', patience=5)

    # Fine-tune the model on the extrapolation task
    model.fit(X_train_extrap, y_train_extrap, validation_data=(X_test_extrap, y_test_extrap), epochs=20, batch_size=32,
              callbacks=[model_checkpoint_callback_extrap, early_stopping_callback_extrap])

    # Make predictions
    predictions = model.predict(X_test_extrap)

    # Rescale predictions
    predictions = scaler.inverse_transform(
        np.concatenate((predictions, np.zeros((predictions.shape[0], num_features - 1))), axis=1))[:, 0]

    # Prepare full dataset predictions for plotting
    predictions_full = np.zeros((len(stock_data), 1))
    predictions_full[:] = np.nan
    predictions_full[train_size + time_steps:] = predictions.reshape(-1, 1)

    # Predict future stock prices
    last_sequence = scaled_data[-time_steps:]
    future_predictions = []

    for _ in range(future_days):
        next_prediction = model.predict(last_sequence.reshape(1, time_steps, num_features))
        future_predictions.append(next_prediction[0, 0])
        next_prediction_full = np.concatenate((next_prediction, np.zeros((1, num_features - 1))), axis=1)
        last_sequence = np.append(last_sequence[1:], next_prediction_full, axis=0)

    # Convert future_predictions to a NumPy array
    future_predictions = np.array(future_predictions)

    # Rescale future predictions
    future_predictions = scaler.inverse_transform(
        np.concatenate((future_predictions.reshape(-1, 1), np.zeros((future_predictions.shape[0], num_features - 1))),
                       axis=1))[:, 0]

    # Extend the stock_data index for future dates
    last_date = stock_data.index[-1]
    future_dates = pd.date_range(last_date, periods=future_days + 1)[1:]

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], color='blue', label='Actual Stock Price')
    plt.plot(pd.DataFrame(predictions_full, index=stock_data.index), color='orange', label='Predicted Stock Price')
    plt.fill_between(stock_data.index[train_size + time_steps:],
                     stock_data['Close'].values[train_size + time_steps:].flatten(),
                     predictions.flatten(), color='orange', alpha=0.3)
    plt.plot(future_dates, future_predictions, color='red', linestyle='--', label='Future Predictions')
    plt.title(f'{stock_symbol} Stock Price Prediction with CNN-LSTM')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()


# Call the function with the stock symbol, desired test ratio, and future days to predict
stock_market_analysis_with_cnn_lstm('NVDA', test_ratio=0.3, future_days=30)
