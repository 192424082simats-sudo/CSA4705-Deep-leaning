import numpy as np
from sklearn.mixture import GaussianMixture

X = np.array([
    [1, 2], [1.5, 1.8], [2, 2],
    [8, 8], [8.5, 8], [9, 8.5],
    [1, 3], [2, 3], [9, 9]
])

gmm = GaussianMixture(
    n_components=2,
    random_state=42,
    n_init=5
)

gmm.fit(X)

labels = gmm.predict(X)
probabilities = gmm.predict_proba(X)

print("Cluster Labels:")
print(labels)

print("Cluster Probabilities:")
print(np.round(probabilities, 2))

print("GMM Clustering Completed")
