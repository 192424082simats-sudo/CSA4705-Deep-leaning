import numpy as np

actual = np.array([1, 0, 1, 0, 1])
predicted = np.array([0.90, 0.20, 0.70, 0.40, 0.60])

predicted = np.clip(predicted, 1e-15, 1 - 1e-15)

loss = -np.mean(
    actual * np.log(predicted)
    + (1 - actual) * np.log(1 - predicted)
)

print("Actual Values:", actual)
print("Predicted Values:", predicted)
print("Cross Entropy Loss:", loss)
