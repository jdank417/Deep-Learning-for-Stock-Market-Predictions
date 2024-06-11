import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define the function to create the model
def create_model(lstm_units=50, conv_filters=32, conv_kernel_size=2, dropout_rate=0.2):
    model = Sequential()
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation='relu', input_shape=(time_steps, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define the function to perform stock market analysis with CNN-LSTM
def stock_market_analysis_with_cnn_lstm(stock_symbol, test_ratio=0.2, future_days=30):
    # Download stock data
    stock_data = yf.download(stock_symbol, start='2010-01-01', end='2024-01-01')
    stock_data = stock_data[['Close']]

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)

    # Prepare the dataset
    global time_steps
    time_steps = 60

    def create_dataset(data):
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i - time_steps:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data)

    # Split the data into training and testing sets
    train_size = int(len(X) * (1 - test_ratio))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Reshape the data to fit the model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Use Sci-Keras KerasRegressor with GridSearchCV for hyperparameter tuning
    model = KerasRegressor(model=create_model, verbose=1)

    # Define the grid search parameters
    param_grid = {
        'model__lstm_units': [50],
        'model__conv_filters': [32],
        'model__conv_kernel_size': [2],
        'model__dropout_rate': [0.2],
        'epochs': [10],
        'batch_size': [16]
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)

    # Print the best parameters
    print("Best Parameters: ", grid_result.best_params_)

    # Train the model with the best parameters
    best_model = grid_result.best_estimator_.model_

    # Define callbacks
    model_checkpoint_callback = ModelCheckpoint(
        filepath=stock_symbol + "_model/best_model.keras",
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

    # Train the best model
    history = best_model.fit(
        X_train, y_train,
        epochs=grid_result.best_params_['epochs'],
        batch_size=grid_result.best_params_['batch_size'],
        validation_data=(X_test, y_test),
        callbacks=[model_checkpoint_callback, early_stopping_callback]
    )

    # Make predictions
    predictions = best_model.predict(X_test)

    # Rescale predictions
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Prepare full dataset predictions for plotting
    predictions_full = np.zeros((len(stock_data), 1))
    predictions_full[:] = np.nan
    predictions_full[train_size + time_steps:] = predictions

    # Predict future stock prices
    last_sequence = scaled_data[-time_steps:]
    future_predictions = []

    for _ in range(future_days):
        next_prediction = best_model.predict(last_sequence.reshape(1, time_steps, 1))
        future_predictions.append(next_prediction[0, 0])
        last_sequence = np.append(last_sequence[1:], next_prediction)

    # Rescale future predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Extend the stock_data index for future dates
    last_date = stock_data.index[-1]
    future_dates = pd.date_range(last_date, periods=future_days + 1)[1:]

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data, color='blue', label='Actual Stock Price')
    plt.plot(pd.DataFrame(predictions_full, index=stock_data.index), color='orange', label='Predicted Stock Price')
    plt.fill_between(stock_data.index[train_size + time_steps:], stock_data.values[train_size + time_steps:].flatten(), predictions.flatten(), color='orange', alpha=0.3)
    plt.plot(future_dates, future_predictions, color='red', linestyle='--', label='Future Predictions')
    plt.title(f'{stock_symbol} Stock Price Prediction with CNN-LSTM')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function with the stock symbol, desired test ratio, and future days to predict
stock_market_analysis_with_cnn_lstm('NVDA', test_ratio=0.1, future_days=150)