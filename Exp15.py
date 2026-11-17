import numpy as np

X = np.arange(1, 11, dtype=float)
y = 2 * X + 1

learning_rate = 0.01
epochs = 30

def update(w, b, x, y):
    error = w * x + b - y
    dw = 2 * np.mean(error * x)
    db = 2 * np.mean(error)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    return w, b

# Batch Gradient Descent
w, b = 0, 0

for i in range(epochs):
    w, b = update(w, b, X, y)

print("Batch GD:", w, b)

# Stochastic Gradient Descent
w, b = 0, 0

for i in range(epochs):
    for j in range(len(X)):
        w, b = update(w, b, X[j:j+1], y[j:j+1])

print("Stochastic GD:", w, b)

# Mini-Batch Gradient Descent
w, b = 0, 0

for i in range(epochs):
    for j in range(0, len(X), 3):
        w, b = update(w, b, X[j:j+3], y[j:j+3])

print("Mini-Batch GD:", w, b)
