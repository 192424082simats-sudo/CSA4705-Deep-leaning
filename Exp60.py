
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

tf.keras.utils.set_random_seed(42)

# Generate temperature data
data = 25 + 5 * np.sin(np.arange(500) * 0.05)

def train_rnn(sequence_length):

    X, y = [], []

    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])

    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y)

    split = int(0.8 * len(X))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        tf.keras.Input(shape=(sequence_length, 1)),
        SimpleRNN(16),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=10, verbose=0)

    return model.evaluate(X_test, y_test, verbose=0)

print("Sequence Length Comparison")

for length in [5, 10, 20, 30]:
    loss = train_rnn(length)
    print("Sequence Length:", length, "Loss:", round(loss, 5))

print("Experiment Completed")
