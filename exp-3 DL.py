import numpy as np
X = np.array([[30, 60],
              [35, 70]])
W = np.array([[0.2, 0.5],
              [0.4, 0.3]])
Y = np.dot(X, W)
print("Input Matrix:")
print(X)
print("Weight Matrix:")
print(W)
print("Predicted Output Matrix:")
print(Y)
