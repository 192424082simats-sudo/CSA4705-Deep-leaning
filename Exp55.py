
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

tf.keras.utils.set_random_seed(42)

# Generate sample stock price data
data = 100 + np.cumsum(np.random.normal(0, 1, 500))

# Normalize
minimum = data.min()
maximum = data.max()
data = (data - minimum) / (maximum - minimum)

# Create sequences
X, y = [], []

for i in range(10, len(data)):
    X.append(data[i-10:i])
    y.append(data[i])

X = np.array(X).reshape(-1, 10, 1)
y = np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# LSTM model
model = Sequential([
    tf.keras.Input(shape=(10, 1)),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=10, verbose=0)

pred = model.predict(X_test, verbose=0).flatten()

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)

print("LSTM Stock Prediction")
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("Prediction Completed")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

tf.keras.utils.set_random_seed(42)

# Generate weather data
time = np.arange(500)

temperature = 25 + 5 * np.sin(time * 0.05)
humidity = 60 + 10 * np.cos(time * 0.05)
pressure = 1010 + 5 * np.sin(time * 0.03)

data = np.column_stack([
    temperature, humidity, pressure
])

# Normalize each feature
minimum = data.min(axis=0)
maximum = data.max(axis=0)
data = (data - minimum) / (maximum - minimum)

# Create sequences
X, y = [], []

for i in range(10, len(data)):
    X.append(data[i-10:i])
    y.append(data[i])

X = np.array(X)
y = np.array(y)

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# GRU model
model = Sequential([
    tf.keras.Input(shape=(10, 3)),
    GRU(32),
    Dense(3)
])

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=15, verbose=0)

prediction = model.predict(X_test[:3], verbose=0)

print("Weather Forecasting")

for i, row in enumerate(prediction):
    print("Day", i+1, ":", np.round(row, 4))

print("GRU Forecasting Completed")
