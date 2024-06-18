import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Input, Reshape, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import logging

# Create a 'logs' directory if it doesn't exist
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(filename=os.path.join(log_dir, 'stock_prediction.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stock_prediction')

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
    data = data.copy()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)
    return data

def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, :])
        y.append(data[i, 0])  # Predict 'Close' price
    return np.array(X), np.array(y)

def build_gan(latent_dim, num_features):
    # Generator
    gen_input = Input(shape=(latent_dim,))
    gen = Dense(128, activation='relu')(gen_input)
    gen = Dense(256, activation='relu')(gen)
    gen = Dense(512, activation='relu')(gen)
    gen_output = Dense(num_features, activation='linear')(gen)
    generator = Model(gen_input, gen_output)

    # Discriminator
    disc_input = Input(shape=(num_features,))
    disc = Dense(512, activation='relu')(disc_input)
    disc = Dense(256, activation='relu')(disc)
    disc = Dense(128, activation='relu')(disc)
    disc_output = Dense(1, activation='sigmoid')(disc)
    discriminator = Model(disc_input, disc_output)

    # Combined GAN model
    gan_input = Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)

    discriminator.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')

    return generator, discriminator, gan

def generate_fake_samples(generator, latent_dim, num_samples):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_data = generator.predict(noise)
    return generated_data

def train_gan(generator, discriminator, gan, real_data, latent_dim, epochs, batch_size):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, real_data.shape[0], batch_size)
        real_samples = real_data[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_samples = generator.predict(noise)

        disc_loss_real = discriminator.train_on_batch(real_samples, valid)
        disc_loss_fake = discriminator.train_on_batch(fake_samples, fake)
        disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

        # Train generator (via GAN model)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_loss = gan.train_on_batch(noise, valid)

        # Print progress
        print(f"Epoch {epoch + 1}, [Discriminator Loss: {disc_loss[0]}, Acc.: {100 * disc_loss[1]:.2f}%], Generator Loss: {gen_loss}")

def augment_data_with_gan(real_data, generator, latent_dim, num_samples):
    fake_data = generate_fake_samples(generator, latent_dim, num_samples)
    augmented_data = np.concatenate((real_data, fake_data), axis=0)
    np.random.shuffle(augmented_data)
    return augmented_data

def stock_market_analysis_with_gan(stock_symbol, test_ratio, future_days):
    logger.info(f"Starting analysis for stock symbol: {stock_symbol}")
    logger.info(f"Test ratio: {test_ratio}, Future days: {future_days}")

    # Download stock data
    logger.info(f"Downloading stock data for {stock_symbol}")
    stock_data = yf.download(stock_symbol, start='2010-01-01', end='2024-01-01')
    logger.info(f"Downloaded data shape: {stock_data.shape}")

    stock_data = stock_data[['Close', 'Volume']]

    if stock_data.empty:
        logger.error("Failed to retrieve data for the given stock symbol")
        raise ValueError("Failed to retrieve data for the given stock symbol")

    # Add technical indicators
    logger.info("Adding technical indicators")
    stock_data = add_technical_indicators(stock_data)
    logger.info(f"Technical indicators added. Updated data shape: {stock_data.shape}")

    # Scale the data
    logger.info("Scaling the data")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)
    logger.info("Data scaled successfully")

    # Prepare the dataset
    logger.info("Preparing dataset")
    time_steps = 60
    num_features = scaled_data.shape[1]

    # Initialize GAN parameters
    latent_dim = 100
    num_samples = 1000
    epochs_gan = 100
    batch_size_gan = 32

    # Build GAN
    generator, discriminator, gan = build_gan(latent_dim, num_features=num_features)

    # Train GAN on real stock data
    train_gan(generator, discriminator, gan, scaled_data, latent_dim, epochs_gan, batch_size_gan)

    # Generate synthetic data with GAN
    synthetic_data = generate_fake_samples(generator, latent_dim, num_samples)

    # Augment real data with synthetic data
    augmented_data = augment_data_with_gan(scaled_data, generator, latent_dim, num_samples)

    # Create the dataset with augmented data
    X, y = create_dataset(augmented_data, time_steps)

    # Split the data into training and testing sets
    train_size = int(len(X) * (1 - test_ratio))

    # Use 90% for training
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Define callbacks for extrapolation
    model_checkpoint_callback_extra = ModelCheckpoint(
        filepath='Utils/extrapolation_model_weights.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback_extra = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model on the extrapolation task
    logger.info("Training model on extrapolation task")
    model = create_model(time_steps=time_steps, num_features=num_features)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32,
              callbacks=[model_checkpoint_callback_extra, early_stopping_callback_extra])

    # Make predictions
    logger.info("Making predictions")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], num_features - 1))), axis=1))[:, 0]

    # Prepare for future predictions
    logger.info("Preparing future predictions")
    last_sequence = scaled_data[-time_steps:]
    future_predictions = []
    for _ in range(future_days):
        next_pred = model.predict(last_sequence.reshape(1, time_steps, num_features))
        next_pred_rescaled = scaler.inverse_transform(np.concatenate((next_pred, np.zeros((1, num_features - 1))), axis=1))[:, 0]
        future_predictions.append(next_pred_rescaled[0])
        next_pred_full = np.concatenate((next_pred, np.zeros((1, num_features - 1))), axis=1)  # Adjust shape
        last_sequence = np.append(last_sequence[1:], next_pred_full, axis=0)

    # Inverse transform the actual prices for plotting
    logger.info("Inverse transforming the actual prices for plotting")
    actual_data = scaler.inverse_transform(scaled_data)
    actual_prices = actual_data[:, 0]

    # Prepare predictions for plotting
    logger.info("Preparing predictions for plotting")
    predictions_full = np.empty_like(actual_prices)
    predictions_full[:] = np.nan
    predictions_full[train_size + time_steps: train_size + time_steps + len(predictions)] = predictions

    future_predictions_full = np.empty_like(actual_prices)
    future_predictions_full[:] = np.nan
    future_predictions_full[-future_days:] = future_predictions

    # Plot results
    logger.info("Plotting results")
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Stock Price')
    plt.plot(predictions_full, label='Predicted Stock Price')
    plt.plot(future_predictions_full, label='Future Predictions', linestyle='--', color='r')
    plt.title(f'{stock_symbol} Stock Price Prediction with GAN-Augmented LSTM')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{stock_symbol}_stock_price_prediction.png')
    plt.show()

    logger.info("Analysis completed successfully")

# Example usage
stock_market_analysis_with_gan('NVDA', test_ratio=0.2, future_days=90)
