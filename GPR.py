import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import GPy
import matplotlib.dates as mdates

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
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def visualize_results_with_future(data, y_true, y_pred, sigma, future_dates, future_predictions, future_sigma):
    plt.figure(figsize=(20, 10))

    dates = data.index[-len(y_true):]

    # Plot Actual vs Predicted Prices
    plt.subplot(1, 2, 1)
    plt.plot(dates, y_true, label='Actual', marker='o', linestyle='-', color='blue', markersize=5)
    plt.plot(dates[:len(y_pred)], y_pred, label='Predicted', marker='x', linestyle='--', color='orange', markersize=5)
    plt.plot(future_dates, future_predictions, label='Future Predictions', marker='s', linestyle='--', color='green', markersize=5)
    plt.title('Actual vs Predicted Prices', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Price', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Plot Predicted Price with Uncertainty Bands
    plt.subplot(1, 2, 2)
    plt.fill_between(dates[:len(y_pred)], y_pred.flatten() - 2 * np.sqrt(sigma.flatten()),
                     y_pred.flatten() + 2 * np.sqrt(sigma.flatten()), color='gray', alpha=0.2, label='Uncertainty Band')
    plt.fill_between(future_dates, future_predictions - 2 * np.sqrt(future_sigma),
                     future_predictions + 2 * np.sqrt(future_sigma), color='lightgreen', alpha=0.2, label='Future Uncertainty Band')
    plt.plot(dates[:len(y_pred)], y_pred.flatten(), label='Predicted', marker='x', linestyle='-', color='orange', markersize=5)
    plt.plot(future_dates, future_predictions, label='Future Predictions', marker='s', linestyle='-', color='green', markersize=5)
    plt.title('Predicted Price with Uncertainty Bands', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Price', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.show()

def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return mse, mae, r2

def download_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    if stock_data.empty or len(stock_data) < 60:
        raise ValueError("Failed to retrieve sufficient data for the given stock symbol")
    return stock_data[['Close', 'Volume']]

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def train_gaussian_process(X_train, y_train):
    kernel = GPy.kern.RBF(input_dim=X_train.shape[-1] * X_train.shape[-2], ARD=True)
    model_gp = GPy.models.SparseGPRegression(X_train.reshape(X_train.shape[0], -1), y_train.reshape(-1, 1), kernel)
    model_gp.optimize(messages=True, max_f_eval=100)
    return model_gp

def predict_with_gp(model, X):
    y_pred, sigma = model.predict(X.reshape(X.shape[0], -1))
    return y_pred, sigma

def inverse_scale_predictions(predictions, scaler, n_features):
    dummy = np.zeros((predictions.shape[0], n_features))
    dummy[:, 0] = predictions.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

def create_future_dates(last_date, days):
    return pd.date_range(start=last_date, periods=days+1, inclusive='right')

def stock_market_analysis_with_gp(stock_symbol, test_ratio, future_days):
    start_date = '2015-01-01'
    end_date = '2024-01-01'

    stock_data = download_stock_data(stock_symbol, start_date, end_date)
    stock_data = add_technical_indicators(stock_data)
    scaled_data, scaler = scale_data(stock_data)

    time_steps = 30
    X, y = create_dataset(scaled_data, time_steps)

    train_size = int(len(X) * (1 - test_ratio))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    model_gp = train_gaussian_process(X_train, y_train)

    y_pred_train, sigma_train = predict_with_gp(model_gp, X_train)
    y_pred_test, sigma_test = predict_with_gp(model_gp, X_test)

    # Predict future prices
    future_dates = create_future_dates(stock_data.index[-1], future_days)
    last_data = scaled_data[-time_steps:]
    future_predictions = []
    future_sigmas = []

    for _ in range(future_days):
        last_data_reshaped = last_data.reshape(1, time_steps, -1)
        future_pred, future_sigma = predict_with_gp(model_gp, last_data_reshaped)
        future_predictions.append(future_pred[0, 0])
        future_sigmas.append(future_sigma[0, 0])
        last_data = np.vstack([last_data[1:], np.append(future_pred, last_data[0, 1:])])

    future_predictions = np.array(future_predictions)
    future_sigmas = np.array(future_sigmas)

    n_features = scaled_data.shape[1]
    y_pred_train_inv = inverse_scale_predictions(y_pred_train, scaler, n_features)
    y_pred_test_inv = inverse_scale_predictions(y_pred_test, scaler, n_features)
    future_predictions_inv = inverse_scale_predictions(future_predictions.reshape(-1, 1), scaler, n_features)

    y_train_inv = inverse_scale_predictions(y_train.reshape(-1, 1), scaler, n_features)
    y_test_inv = inverse_scale_predictions(y_test.reshape(-1, 1), scaler, n_features)

    mse_train, mae_train, r2_train = calculate_metrics(y_train_inv, y_pred_train_inv)
    mse_test, mae_test, r2_test = calculate_metrics(y_test_inv, y_pred_test_inv)

    print(f"Training MSE: {mse_train:.4f}, MAE: {mae_train:.4f}, R^2: {r2_train:.4f}")
    print(f"Testing MSE: {mse_test:.4f}, MAE: {mae_test:.4f}, R^2: {r2_test:.4f}")

    visualize_results_with_future(stock_data, np.concatenate([y_train_inv, y_test_inv]),
                                  np.concatenate([y_pred_train_inv, y_pred_test_inv]),
                                  np.concatenate([sigma_train, sigma_test]),
                                  future_dates, future_predictions_inv, future_sigmas)

if __name__ == "__main__":
    stock_market_analysis_with_gp('NVDA', test_ratio=0.2, future_days=90)
