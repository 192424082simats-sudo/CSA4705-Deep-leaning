import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

X = np.random.rand(100, 10, 1)
Y = np.random.rand(100, 1)

rnn = Sequential([
    SimpleRNN(16, input_shape=(10, 1)),
    Dense(1)
])

lstm = Sequential([
    LSTM(16, input_shape=(10, 1)),
    Dense(1)
])

rnn.compile(optimizer='adam', loss='mse')
lstm.compile(optimizer='adam', loss='mse')

rnn.fit(X, Y, epochs=2, verbose=0)
lstm.fit(X, Y, epochs=2, verbose=0)

print("RNN Prediction:", rnn.predict(X[:1], verbose=0))
print("LSTM Prediction:", lstm.predict(X[:1], verbose=0))
print("RNN and LSTM comparison completed")
