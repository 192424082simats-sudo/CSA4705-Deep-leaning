import numpy as np

X = np.array([[0.5, 0.8]])
target = np.array([[1.0]])

W1 = np.array([[0.2, 0.4], [0.3, 0.1]])
W2 = np.array([[0.5], [0.6]])

b1 = np.array([[0.1, 0.2]])
b2 = np.array([[0.1]])

learning_rate = 0.1

# Forward propagation
z1 = np.dot(X, W1) + b1
a1 = 1 / (1 + np.exp(-z1))

z2 = np.dot(a1, W2) + b2
a2 = 1 / (1 + np.exp(-z2))

# Calculate error
error = target - a2

# Backpropagation
delta2 = error * a2 * (1 - a2)
delta1 = np.dot(delta2, W2.T) * a1 * (1 - a1)

# Update weights
W2 += learning_rate * np.dot(a1.T, delta2)
W1 += learning_rate * np.dot(X.T, delta1)

b2 += learning_rate * delta2
b1 += learning_rate * delta1

print("Predicted Output:", a2)
print("Error:", error)
print("Updated W1:\n", W1)
print("Updated W2:\n", W2)
