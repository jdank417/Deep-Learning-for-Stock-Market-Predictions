import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import GPy
import matplotlib.dates as mdates
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from joblib import Parallel, delayed


def add_technical_indicators(data):
    data = data.copy()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = calculate_rsi(data['Close'], window=14)
    data['Volatility'] = data['Close'].rolling(window=20).std()
    data.dropna(inplace=True)
    return data


def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def visualize_results_with_future_and_ci(data, y_true, y_pred, sigma, future_dates, future_predictions, future_sigma,
                                         lower_bound, upper_bound):
    plt.figure(figsize=(20, 10))

    dates = data.index[-len(y_true):]

    # Plot Actual vs Predicted Prices
    plt.subplot(1, 2, 1)
    plt.plot(dates, y_true, label='Actual', marker='o', linestyle='-', color='blue', markersize=5)
    plt.plot(dates[:len(y_pred)], y_pred, label='Predicted', marker='x', linestyle='--', color='orange', markersize=5)
    plt.plot(future_dates, future_predictions, label='Future Predictions', marker='s', linestyle='--', color='green',
             markersize=5)
    plt.fill_between(future_dates, lower_bound, upper_bound, color='lightgreen', alpha=0.3,
                     label='95% Confidence Interval')
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
                     future_predictions + 2 * np.sqrt(future_sigma), color='lightgreen', alpha=0.2,
                     label='Future Uncertainty Band')
    plt.plot(dates[:len(y_pred)], y_pred.flatten(), label='Predicted', marker='x', linestyle='-', color='orange',
             markersize=5)
    plt.plot(future_dates, future_predictions, label='Future Predictions', marker='s', linestyle='-', color='green',
             markersize=5)
    plt.fill_between(future_dates, lower_bound, upper_bound, color='lightgreen', alpha=0.3,
                     label='95% Confidence Interval')
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


def create_composite_kernel(input_dim, time_steps):
    # Simplified kernel with RBF and White noise only
    rbf_kernel = GPy.kern.RBF(input_dim=input_dim, ARD=True)
    white_kernel = GPy.kern.White(input_dim=input_dim)
    combined_kernel = rbf_kernel + white_kernel
    return combined_kernel


def train_gaussian_process(X_train, y_train, kernel):
    n_samples, time_steps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(n_samples, time_steps * n_features)
    model_gp = GPy.models.GPRegression(X_train_reshaped, y_train.reshape(-1, 1), kernel)
    model_gp.Gaussian_noise.variance.constrain_positive(warning=False)
    model_gp.optimize(messages=True, max_f_eval=100)
    return model_gp


def predict_with_gp(model, X):
    n_samples, time_steps, n_features = X.shape
    X_reshaped = X.reshape(n_samples, time_steps * n_features)
    y_pred, sigma = model.predict(X_reshaped)
    return y_pred, sigma


def inverse_scale_predictions(predictions, scaler, n_features):
    dummy = np.zeros((predictions.shape[0], n_features))
    dummy[:, 0] = predictions.flatten()
    return scaler.inverse_transform(dummy)[:, 0]


def create_future_dates(last_date, days):
    return pd.date_range(start=last_date, periods=days + 1, inclusive='right')


def predict_future_prices(model, last_data, future_days, time_steps, scaler, n_features):
    future_predictions = []
    future_sigmas = []
    current_window = last_data.copy()

    for _ in range(future_days):
        current_window_reshaped = current_window.reshape(1, time_steps * n_features)
        future_pred, future_sigma = model.predict(current_window_reshaped)
        future_predictions.append(future_pred[0, 0])
        future_sigmas.append(future_sigma[0, 0])

        new_datapoint = np.zeros((1, n_features))
        new_datapoint[0, 0] = future_pred[0, 0]
        current_window = np.vstack([current_window[1:], new_datapoint])

    future_predictions = np.array(future_predictions)
    future_sigmas = np.array(future_sigmas)

    return inverse_scale_predictions(future_predictions.reshape(-1, 1), scaler, n_features), future_sigmas


def time_series_cv(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = Parallel(n_jobs=-1)(delayed(train_and_evaluate_fold)(X, y, train_index, test_index)
                                    for train_index, test_index in tscv.split(X))

    cv_scores = [score for score in cv_scores if score is not None]
    if len(cv_scores) == 0:
        print("Error: All cross-validation folds failed. The model may not be suitable for this data.")
        return None, None
    return np.mean(cv_scores), np.std(cv_scores)


def train_and_evaluate_fold(X, y, train_index, test_index):
    try:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        kernel = create_composite_kernel(input_dim=X_train.shape[1], time_steps=X_train.shape[1] // X_train.shape[2])
        model_gp = train_gaussian_process(X_train, y_train, kernel)

        y_pred, _ = predict_with_gp(model_gp, X_test)
        mse, _, _ = calculate_metrics(y_test, y_pred)
        return mse
    except np.linalg.LinAlgError:
        print("Warning: LinAlgError occurred during cross-validation. Skipping this fold.")
        return None


def stock_market_analysis_with_gp(stock_symbol, start_date='2000-01-01', end_date='2023-12-31', test_ratio=0.2, future_days=90):
    data = download_stock_data(stock_symbol, start_date, end_date)
    data_with_indicators = add_technical_indicators(data)
    scaled_data, scaler = scale_data(data_with_indicators)

    time_steps = 20
    n_features = scaled_data.shape[1]
    X, y = create_dataset(scaled_data, time_steps)

    test_size = int(len(X) * test_ratio)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    kernel = create_composite_kernel(input_dim=X_train.shape[1], time_steps=time_steps)

    cv_score_mean, cv_score_std = time_series_cv(X_train, y_train, n_splits=5)
    if cv_score_mean is None:
        print("Cross-validation failed. Proceeding with caution.")
    else:
        print(f"Cross-Validation Mean MSE: {cv_score_mean:.4f}, Std: {cv_score_std:.4f}")

    model_gp = train_gaussian_process(X_train, y_train, kernel)

    y_pred, sigma = predict_with_gp(model_gp, X_test)
    mse, mae, r2 = calculate_metrics(y_test, y_pred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

    y_true_inverse = inverse_scale_predictions(y_test, scaler, n_features)
    y_pred_inverse = inverse_scale_predictions(y_pred, scaler, n_features)

    future_dates = create_future_dates(data_with_indicators.index[-1], future_days)
    last_data = scaled_data[-time_steps:]
    future_predictions, future_sigma = predict_future_prices(model_gp, last_data, future_days, time_steps, scaler,
                                                             n_features)

    lower_bound = future_predictions - 2 * np.sqrt(future_sigma)
    upper_bound = future_predictions + 2 * np.sqrt(future_sigma)

    visualize_results_with_future_and_ci(data_with_indicators, y_true_inverse, y_pred_inverse, sigma, future_dates,
                                         future_predictions, future_sigma, lower_bound, upper_bound)


if __name__ == "__main__":
    stock_market_analysis_with_gp('NVDA', test_ratio=0.2, future_days=90)
