import numpy as np

X = np.array([[0.5, 0.8]])

W1 = np.array([
    [0.2, 0.4],
    [0.3, 0.1]
])

b1 = np.array([0.1, 0.2])

W2 = np.array([
    [0.5],
    [0.6]
])

b2 = np.array([0.1])

# Hidden layer
hidden = np.maximum(0, np.dot(X, W1) + b1)

# Output layer
output = 1 / (1 + np.exp(-(np.dot(hidden, W2) + b2)))

print("Input:", X)
print("Hidden Layer Output:", hidden)
print("Final Output:", output)
