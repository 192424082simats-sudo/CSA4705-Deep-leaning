import numpy as np
A = np.array([2, 4, 6])
B = np.array([1, 3, 5])
combined = A + B
similarity = np.dot(A, B)
print("First Movie Vector:", A)
print("Second Movie Vector:", B)
print("Combined Representation:", combined)
print("Dot Product:", similarity)
