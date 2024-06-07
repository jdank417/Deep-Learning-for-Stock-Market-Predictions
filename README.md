**Stock Market Prediction Program**
Welcome to the Stock Market Prediction Program!

This project leverages machine learning techniques to predict stock market movements using tools and libraries including TensorFlow, Scikit-learn, and various technical analysis indicators. The program retrieves historical stock data, preprocesses it, and applies an LSTM neural network for predictions.

Features
Stock Data Retrieval: Downloads historical stock data using the yfinance library.
Technical Indicators: Calculates RSI, MACD, Bollinger Bands, and moving averages.
Data Visualization: Plots historical closing prices and predictions.
Risk Analysis: Computes annualized volatility and Sharpe ratio.
LSTM Model: Implements an LSTM neural network with hyperparameter tuning using Bayesian optimization.
Model Persistence: Saves the best model and predictions for future use.
Requirements
Ensure you have the following prerequisites installed:

Python 3.8.0 (64-bit)
Required libraries: see document req_lib

You can install the required libraries using pip:
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow scikeras ta skopt

**Implementation Details**
The script performs the following steps:

Data Retrieval:

Downloads historical stock data and benchmark data.
Fills missing values and calculates additional features like rate of return, RSI, MACD, Bollinger Bands, and moving averages.
Data Visualization:

Plots historical closing prices.
Risk Analysis:

Calculates annualized volatility and Sharpe ratio.
Data Preparation:

Scales the data using MinMaxScaler.
Splits the data into training and testing sets.
Model Definition:

Defines an LSTM neural network model using TensorFlow and Keras.
Hyperparameter Tuning:

It uses Bayesian optimization to tune hyperparameters.
Model Training:

Trains the final model with the best hyperparameters.
Saves the model architecture and weights.
Predictions:

Makes predictions on the test set.
Plots the predicted vs. actual prices.
Notes
It is recommended to use the 64-bit version of Python 3.8.0 to avoid potential compatibility issues.
The hyperparameter tuning process can be time-consuming and may require significant computational resources.


