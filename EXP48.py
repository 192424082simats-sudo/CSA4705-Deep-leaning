import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Daily temperature data
temperature = np.array([
    25, 26, 27, 28, 29, 30, 31, 30, 29, 28,
    27, 26, 25, 26, 27, 28, 29, 30, 31, 32,
    31, 30, 29, 28, 27, 26, 25, 26, 27, 28
], dtype=float)

# Normalize data
minimum = temperature.min()
maximum = temperature.max()

data = (temperature - minimum) / (maximum - minimum)

X = []
y = []

for i in range(len(data)-5):
    X.append(data[i:i+5])
    y.append(data[i+5])

X = np.array(X).reshape(-1,5,1)
y = np.array(y)

model = Sequential([
    SimpleRNN(32, activation='tanh',
              input_shape=(5,1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=100, verbose=0)

# Predict next temperature
last_values = data[-5:].reshape(1,5,1)

prediction = model.predict(last_values)

predicted_temperature = (
    prediction[0][0] * (maximum-minimum) + minimum
)

print("Last 5 Days Temperature:", temperature[-5:])
print("Predicted Next Day Temperature:",
      round(float(predicted_temperature), 2))
