import numpy as np

A = np.array([2, 4, 6])
B = np.array([5, 8, 6])

euclidean = np.linalg.norm(A - B)
manhattan = np.sum(np.abs(A - B))

print("Euclidean Distance:", euclidean)
print("Manhattan Distance:", manhattan)
