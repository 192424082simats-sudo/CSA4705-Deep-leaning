import numpy as np

P = np.array([0.5, 0.3, 0.2])
Q = np.array([0.4, 0.4, 0.2])

kl = np.sum(P * np.log(P / Q))

print("Actual Distribution:", P)
print("Predicted Distribution:", Q)
print("KL Divergence:", kl)
