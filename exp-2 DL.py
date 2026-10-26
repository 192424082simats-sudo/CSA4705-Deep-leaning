import numpy as np
X = np.array([4, 6, 8])
W = np.array([0.5, 0.3, 0.2])
b = 2
dot_product = np.dot(X, W)
Y = dot_product + b
print("Input Features:", X)
print("Weights:", W)
print("Dot Product:", dot_product)
print("Bias:", b)
print("Predicted Output:", Y)
