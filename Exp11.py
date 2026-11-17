import numpy as np

data = ["Low", "Low", "Low", "High",
        "High", "Low", "High", "Low"]

values, counts = np.unique(data, return_counts=True)

probabilities = counts / len(data)

entropy = -np.sum(
    probabilities * np.log2(probabilities)
)

print("Class Values:", values)
print("Probabilities:", probabilities)
print("Entropy:", entropy)
