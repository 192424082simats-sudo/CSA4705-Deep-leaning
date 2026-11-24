
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

tf.keras.utils.set_random_seed(42)

# Generate sample sales data
data = 100 + 20 * np.sin(np.arange(500) * 0.05)
data = (data - data.min()) / (data.max() - data.min())

X, y = [], []

for i in range(10, len(data)):
    X.append(data[i-10:i])
    y.append(data[i])

X = np.array(X).reshape(-1, 10, 1)
y = np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def train_model(layer):
    model = Sequential([
        tf.keras.Input(shape=(10, 1)),
        layer,
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=10, verbose=0)

    return model.evaluate(X_test, y_test, verbose=0)

rnn_loss = train_model(SimpleRNN(32))
lstm_loss = train_model(LSTM(32))

print("RNN Test Loss:", round(rnn_loss, 4))
print("LSTM Test Loss:", round(lstm_loss, 4))

print("Model Comparison Completed")
