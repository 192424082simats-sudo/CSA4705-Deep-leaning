import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

X = np.array([
    [10, 20, 30, 40],
    [20, 30, 40, 50],
    [30, 40, 50, 60],
    [40, 50, 60, 70],
    [50, 60, 70, 80],
    [60, 70, 80, 90]
])

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_reconstructed = pca.inverse_transform(X_pca)

fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X_scaled)

fa_reconstructed = X_fa @ fa.components_ + fa.mean_

pca_error = np.mean((X_scaled - pca_reconstructed) ** 2)
fa_error = np.mean((X_scaled - fa_reconstructed) ** 2)

print("PCA Reduced Shape:", X_pca.shape)
print("FA Reduced Shape:", X_fa.shape)
print("PCA Reconstruction Error:", round(pca_error, 4))
print("FA Reconstruction Error:", round(fa_error, 4))
