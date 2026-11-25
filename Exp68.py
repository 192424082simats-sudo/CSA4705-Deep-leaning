from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original Dataset Shape:", X.shape)
print("Reduced Dataset Shape:", X_pca.shape)
print("Explained Variance Ratio:",
      pca.explained_variance_ratio_)
print("Total Variance Retained:",
      sum(pca.explained_variance_ratio_))
