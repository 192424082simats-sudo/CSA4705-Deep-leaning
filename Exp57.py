
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

tf.keras.utils.set_random_seed(42)

# Generate sequential dataset
data = np.sin(np.arange(500) * 0.05)

X, y = [], []

for i in range(10, len(data)):
    X.append(data[i-10:i])
    y.append(data[i])

X = np.array(X).reshape(-1, 10, 1)
y = np.array(y)

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def evaluate_model(layer):
    model = Sequential([
        tf.keras.Input(shape=(10, 1)),
        layer,
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=16,
        verbose=0
    )

    return model.evaluate(X_test, y_test, verbose=0)

rnn = evaluate_model(SimpleRNN(32))
lstm = evaluate_model(LSTM(32))
gru = evaluate_model(GRU(32))

print("Model Performance Comparison")
print("RNN Loss:", round(rnn, 5))
print("LSTM Loss:", round(lstm, 5))
print("GRU Loss:", round(gru, 5))

print("Comparison Completed")
