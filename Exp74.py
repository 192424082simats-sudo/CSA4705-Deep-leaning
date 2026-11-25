import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

X = np.random.randn(200, 4)

X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

ica = FastICA(n_components=2, random_state=42)
X_ica = ica.fit_transform(X)

print("Original Data Shape:", X.shape)
print("PCA Reduced Shape:", X_pca.shape)
print("ICA Reduced Shape:", X_ica.shape)
print("PCA Explained Variance:",
      np.round(pca.explained_variance_ratio_, 3))
print("PCA and ICA Comparison Completed")
