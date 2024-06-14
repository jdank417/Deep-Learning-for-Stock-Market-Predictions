import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import GPy

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

def visualize_results(data, y_true, y_pred, sigma):
    plt.figure(figsize=(14, 7))
    dates = data.index[-len(y_true):]

    plt.subplot(1, 2, 1)
    plt.plot(dates, y_true, label='Actual', marker='o')
    plt.plot(dates[:len(y_pred)], y_pred, label='Predicted', marker='x')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.fill_between(dates[:len(y_pred)], y_pred.flatten() - 2 * np.sqrt(sigma.flatten()),
                     y_pred.flatten() + 2 * np.sqrt(sigma.flatten()), alpha=0.2)
    plt.plot(dates[:len(y_pred)], y_pred.flatten(), label='Predicted', marker='x')
    plt.title('Predicted Price with Uncertainty Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')

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

def stock_market_analysis_with_gp(stock_symbol, test_ratio, future_days):
    start_date = '2015-01-01'  # Reduced dataset size
    end_date = '2024-01-01'

    stock_data = download_stock_data(stock_symbol, start_date, end_date)
    stock_data = add_technical_indicators(stock_data)
    scaled_data, scaler = scale_data(stock_data)

    time_steps = 30  # Reduced look-back period
    X, y = create_dataset(scaled_data, time_steps)

    train_size = int(len(X) * (1 - test_ratio))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    model_gp = train_gaussian_process(X_train, y_train)

    y_pred_train, sigma_train = predict_with_gp(model_gp, X_train)
    y_pred_test, sigma_test = predict_with_gp(model_gp, X_test)

    n_features = scaled_data.shape[1]
    y_pred_train_inv = inverse_scale_predictions(y_pred_train, scaler, n_features)
    y_pred_test_inv = inverse_scale_predictions(y_pred_test, scaler, n_features)

    y_train_inv = inverse_scale_predictions(y_train.reshape(-1, 1), scaler, n_features)
    y_test_inv = inverse_scale_predictions(y_test.reshape(-1, 1), scaler, n_features)

    mse_train, mae_train, r2_train = calculate_metrics(y_train_inv, y_pred_train_inv)
    mse_test, mae_test, r2_test = calculate_metrics(y_test_inv, y_pred_test_inv)

    print(f"Training MSE: {mse_train:.4f}, MAE: {mae_train:.4f}, R^2: {r2_train:.4f}")
    print(f"Testing MSE: {mse_test:.4f}, MAE: {mae_test:.4f}, R^2: {r2_test:.4f}")

    visualize_results(stock_data, np.concatenate([y_train_inv, y_test_inv]),
                      np.concatenate([y_pred_train_inv, y_pred_test_inv]),
                      np.concatenate([sigma_train, sigma_test]))

if __name__ == "__main__":
    stock_market_analysis_with_gp('NVDA', test_ratio=0.2, future_days=30)
