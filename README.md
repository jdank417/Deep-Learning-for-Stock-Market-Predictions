**Stock Market Analysis with CNN-LSTM**
This project implements a Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) model to predict stock prices. The model uses historical stock data, along with technical indicators, to predict future stock prices.

**Features**
Data Retrieval: Fetches historical stock data using the yfinance library.
Technical Indicators: Adds moving averages (MA20, MA50) as features to the dataset.
Data Preprocessing: Scales the data using MinMaxScaler and creates datasets for model training.
Model Architecture: Combines CNN and LSTM layers to capture both spatial and temporal dependencies in the data.
Training Strategy: Pre-trains the model for interpolation, then fine-tunes it for extrapolation.
Predictions: Predict both historical stock prices and future stock prices.

**Installation**
Clone the Repository

Clone the repository from GitHub and navigate into the project directory.

Install Dependencies

Ensure Python is installed on your system. Install the required packages using a package manager like pip.

**Usage**
Import Required Modules

Import libraries such as yfinance for data retrieval, numpy and pandas for data manipulation, matplotlib for plotting, scikit-learn for data preprocessing, and tensorflow for building and training the model.

Define Model Architecture

Create a model that integrates CNN and LSTM layers, enabling the capture of spatial features from the convolutional layers and temporal dependencies from the LSTM layers.

Add Technical Indicators

Enhance the dataset by adding moving averages (MA20 and MA50) to provide additional features that help the model understand market trends.

Data Preprocessing

Scale the data using MinMaxScaler to normalize the feature values. Create datasets for model training, including input sequences and corresponding target values.

**Training Strategy**

Interpolation: Pre-train the model to predict within the known historical data. Split the data into training and validation sets for this purpose. Use callbacks for early stopping and model checkpointing to optimize training.
Extrapolation: Fine-tune the model to predict future values. Load pre-trained weights from the interpolation step and train the model on a dataset split into training and test sets.
Predictions

Make predictions on the test dataset to evaluate the model's performance on historical data. Additionally, generate future stock price predictions using the trained model.

**Plotting Results**

Visualize the actual and predicted stock prices using line plots. Highlight future predictions to differentiate them from historical predictions.

**Example**
To run the stock market analysis, specify the stock symbol, the ratio of test data, and the number of future days to predict. The function will download the stock data, preprocess it, train the model, and generate predictions. The results will be plotted for easy visualization.
