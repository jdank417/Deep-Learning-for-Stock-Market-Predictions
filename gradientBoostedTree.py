#Implmentation of Gradient Boosted Trees for Stock Price Prediction

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas_datareader.data as web
import requests

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# Function to fetch news data
def fetch_news_data(start_date, end_date):
    api_key = '76d8071e2990471a9e992c09a806410e'  # Replace with your actual NewsAPI API key
    url = f'https://newsapi.org/v2/everything?q=stock+market&from={start_date}&to={end_date}&sortBy=publishedAt&apiKey={api_key}'
    response = requests.get(url)
    data = response.json()

    if data['status'] == 'ok':
        articles = data['articles']
        news_data = pd.DataFrame(
            [{'date': article['publishedAt'][:10], 'text': article['description']} for article in articles])
        news_data['date'] = pd.to_datetime(news_data['date'])
        return news_data
    else:
        print(f"Error: {data['message']}")
        return pd.DataFrame()


# Function to get macroeconomic indicators
def get_macro_indicators(start_date, end_date):
    gdp = web.DataReader('GDP', 'fred', start_date, end_date)
    inflation = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)
    interest_rate = web.DataReader('DFF', 'fred', start_date, end_date)
    return gdp, inflation, interest_rate


# Function to get sentiment score
def get_sentiment_score(text):
    scores = sia.polarity_scores(text)
    return scores['compound']


# Add additional features
def add_additional_features(data, start_date, end_date):
    # Sentiment analysis
    today = datetime.now().date()
    thirty_days_ago = today - timedelta(days=30)
    news_start_date = thirty_days_ago
    news_end_date = today

    news_data = fetch_news_data(news_start_date, news_end_date)
    if not news_data.empty and 'text' in news_data.columns:
        # Calculate sentiment scores for each news article
        news_data['Sentiment_Score'] = news_data['text'].apply(lambda x: get_sentiment_score(x) if isinstance(x, str) else 0)
        # Aggregate sentiment scores for the given period (e.g., daily)
        daily_sentiment = news_data.groupby(pd.Grouper(key='date', freq='D'))['Sentiment_Score'].mean().fillna(0)
        # Merge aggregated sentiment scores with stock data
        data = data.merge(daily_sentiment, left_index=True, right_index=True, how='left').fillna(0)
    else:
        data['Sentiment_Score'] = 0

    # Macroeconomic indicators
    gdp, inflation, interest_rate = get_macro_indicators(start_date, end_date)
    data['GDP'] = gdp.resample('D').interpolate(method='linear').reindex(data.index).fillna(method='ffill').fillna(
        method='bfill')
    data['Inflation'] = inflation.resample('D').interpolate(method='linear').reindex(data.index).fillna(
        method='ffill').fillna(method='bfill')
    data['Interest_Rate'] = interest_rate.resample('D').interpolate(method='linear').reindex(data.index).fillna(
        method='ffill').fillna(method='bfill')

    return data



def stock_market_analysis(stock_ticker, start_date="2010-01-01", end_date=None, test_ratio=0.2, benchmark_ticker='VOO'):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Step 1: Retrieve Stock Data
    data = yf.download(stock_ticker, start=start_date, end=end_date)
    benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)

    if data.empty or benchmark_data.empty:
        print("Failed to retrieve data. Check the ticker symbols and network connection.")
        return

    # Fill NaN values with 0
    data['Volume'] = data['Volume'].fillna(0)
    benchmark_data['Volume'] = benchmark_data['Volume'].fillna(0)

    # Calculate additional features
    data['Rate of Return'] = data['Close'].pct_change().fillna(0)
    data['RSI'] = RSIIndicator(data['Close'], window=14).rsi().fillna(0)
    macd = MACD(data['Close'])
    data['MACD'] = macd.macd().fillna(0)
    data['MACD_Signal'] = macd.macd_signal().fillna(0)
    bollinger = BollingerBands(data['Close'])
    data['BB_High'] = bollinger.bollinger_hband().fillna(0)
    data['BB_Low'] = bollinger.bollinger_lband().fillna(0)

    # Calculate Moving Average
    data['50_MA'] = data['Close'].rolling(window=50).mean().fillna(0)

    # Add benchmark closing prices as a new feature
    data['Benchmark_Close'] = benchmark_data['Close']

    # Add additional features
    data = add_additional_features(data, start_date, end_date)

    # List of features
    features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'Rate of Return', '50_MA',
                'Benchmark_Close', 'Sentiment_Score', 'GDP', 'Inflation', 'Interest_Rate']

    # Fill remaining NaN values
    data.fillna(0, inplace=True)

    # Step 2: Data Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label=stock_ticker)
    plt.title(f'Historical Closing Prices of {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Step 3: Risk Analysis
    returns = data['Close'].pct_change()
    volatility = returns.std()
    annual_volatility = volatility * np.sqrt(252)
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

    print(f"Annualized Volatility of {stock_ticker}: {annual_volatility}")
    print(f"Sharpe Ratio of {stock_ticker}: {sharpe_ratio}")

    # Step 4: Prepare Data for Gradient Boosted Trees
    X = data[features].values
    y = data['Close'].shift(-1).fillna(method='ffill').values  # Predict next day's closing price

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Impute missing values using KNN
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_scaled)

    # Splitting data into train and test sets
    train_size = int(len(X_imputed) * (1 - test_ratio))
    x_train, x_test = X_imputed[:train_size], X_imputed[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Step 5: Model Selection and Hyperparameter Tuning
    models = {
        'Ridge': {
            'model': Ridge(),
            'params': {'alpha': [0.1, 1.0, 10.0]}
        },
        'Lasso': {
            'model': Lasso(),
            'params': {'alpha': [0.1, 1.0, 10.0]}  # Remove 'learning_rate' from parameters
        },
        'SVR': {
            'model': SVR(),
            'params': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}
        },
        'RandomForest': {
            'model': RandomForestRegressor(),
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, None]}
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5],
                       'subsample': [0.8, 1.0], 'max_features': ['sqrt', 'log2']}
        }
    }

    scores = {}
    for name, model_params in models.items():
        model = model_params['model']
        params = model_params['params']
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        scores[name] = cross_val_score(best_model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')

    best_model_name = min(scores, key=lambda k: scores[k].mean())
    best_model = models[best_model_name]['model']
    best_params = grid_search.best_params_
    print(f"Best Model: {best_model_name}")
    print(f"Best Parameters: {best_params}")

    # Print additional debugging information
    print(f"Selected Model Object: {best_model}")
    print(f"Valid Parameters for {best_model_name}: {best_model.get_params().keys()}")

    # Print the selected model name
    print("Selected Model:", best_model_name)

    # Print best_params for debugging
    print("Best Parameters:", best_params)

    # Set parameters of the best model
    print("Setting parameters of the best model...")
    best_model.set_params(**best_params)
    print("Parameters set successfully.")

    best_model.fit(x_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(best_model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean Cross-Validation Score: {cv_scores.mean()}")

    # Step 7: Make Predictions
    y_pred_train = best_model.predict(x_train)
    y_pred_test = best_model.predict(x_test)

    # Calculate evaluation metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    print(f"Train MAPE: {train_mape}")
    print(f"Test MAPE: {test_mape}")

    # Step 8: Visualization
    plt.figure(figsize=(14, 7))

    plt.plot(data.index[:train_size], y_train, label='Actual Train Prices')
    plt.plot(data.index[train_size:], y_test, label='Actual Test Prices')
    plt.plot(data.index[:train_size], y_pred_train, label='Predicted Train Prices')
    plt.plot(data.index[train_size:], y_pred_test, label='Predicted Test Prices')

    plt.title(f'Stock Price Prediction of {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# Example usage:
stock_market_analysis('NVDA', test_ratio=0.2)
