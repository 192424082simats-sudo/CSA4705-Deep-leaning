from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data

X_scaled = StandardScaler().fit_transform(X)

pca = PCA()
pca.fit(X_scaled)

variance = pca.explained_variance_ratio_
cumulative = variance.cumsum()

print("Explained Variance:", variance)
print("Cumulative Variance:", cumulative)

components = next(
    i + 1 for i, value in enumerate(cumulative)
    if value >= 0.95
)

print("Components Required for 95% Variance:", components)
