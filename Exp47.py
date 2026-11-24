import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate sensor readings
data = np.sin(np.arange(1000) * 0.05)

X = []
y = []

for i in range(10, len(data)):
    X.append(data[i-10:i])
    y.append(data[i])

X = np.array(X).reshape(-1,10,1)
y = np.array(y)

model = Sequential([
    SimpleRNN(32, activation='tanh',
              input_shape=(10,1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=10, batch_size=32)

prediction = model.predict(X[:5])

print("Actual Sensor Values:", y[:5])
print("Predicted Sensor Values:",
      prediction.flatten())
