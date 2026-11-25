import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = np.array([
    [25, 120, 80, 70],
    [30, 130, 85, 75],
    [35, 140, 90, 80],
    [40, 150, 95, 85],
    [45, 160, 100, 90],
    [50, 170, 105, 95]
])

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original Shape:", X.shape)
print("Reduced Shape:", X_pca.shape)
print("Transformed Data:\n", np.round(X_pca, 2))
print("Explained Variance:", np.round(
    pca.explained_variance_ratio_, 2))
