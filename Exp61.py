
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

tf.keras.utils.set_random_seed(42)

# Generate product demand data
data = 100 + 20 * np.sin(np.arange(400) * 0.05)

X, y = [], []

for i in range(10, len(data)):
    X.append(data[i-10:i])
    y.append(data[i])

X = np.array(X).reshape(-1, 10, 1)
y = np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Test different configurations
configurations = [
    (0.001, 16, 16),
    (0.01, 32, 32),
    (0.0005, 64, 16)
]

print("LSTM Hyperparameter Tuning")

for lr, units, batch in configurations:

    model = Sequential([
        tf.keras.Input(shape=(10, 1)),
        LSTM(units),
        Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss="mse")

    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=batch,
        verbose=0
    )

    loss = model.evaluate(X_test, y_test, verbose=0)

    print("Learning Rate:", lr)
    print("Hidden Units:", units)
    print("Batch Size:", batch)
    print("Test Loss:", round(loss, 4))
    print("-------------------")

print("Hyperparameter Tuning Completed")
