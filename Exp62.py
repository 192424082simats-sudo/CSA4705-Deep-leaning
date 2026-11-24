
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

tf.keras.utils.set_random_seed(42)

# Generate electricity consumption data
data = 100 + 20 * np.sin(np.arange(300) * 0.05)

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
    SimpleRNN(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=15, verbose=0)

# Predictions
predictions = model.predict(X_test, verbose=0).flatten()

# Evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("RNN Model Evaluation")
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))

print("Evaluation Completed")
