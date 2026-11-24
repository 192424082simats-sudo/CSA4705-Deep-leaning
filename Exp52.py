
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

tf.keras.utils.set_random_seed(42)

# Generate electricity consumption data
time = np.arange(500)
data = 200 + 30 * np.sin(time * 0.1) + 10 * np.random.randn(500)

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

# Build RNN model
model = Sequential([
    tf.keras.Input(shape=(10, 1)),
    SimpleRNN(32, activation="tanh"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

# Predict electricity consumption
prediction = model.predict(X_test[:5], verbose=0)

print("Electricity Consumption Forecast")

for i, value in enumerate(prediction):
    print("Day", i+1, ":", round(float(value[0]), 4))

print("Forecasting completed successfully.")
