
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

tf.keras.utils.set_random_seed(42)

# Generate machine sensor data
time = np.arange(500)

temperature = 50 + 10 * np.sin(time * 0.05)
pressure = 100 + 5 * np.cos(time * 0.05)
vibration = 20 + 3 * np.sin(time * 0.1)

data = np.column_stack([
    temperature, pressure, vibration
])

# Normalize
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

# Stacked RNN
model = Sequential([
    tf.keras.Input(shape=(10, 3)),
    SimpleRNN(32, return_sequences=True),
    SimpleRNN(16),
    Dense(3)
])

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=15, verbose=0)

prediction = model.predict(X_test[:3], verbose=0)

print("Machine Condition Prediction")

for i, row in enumerate(prediction):
    print("Prediction", i+1, ":", np.round(row, 4))

print("Stacked RNN Completed")
