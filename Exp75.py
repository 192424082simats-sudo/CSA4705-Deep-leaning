import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

data = np.array([
    [80, 85, 90, 88],
    [70, 75, 80, 78],
    [90, 92, 95, 94],
    [60, 65, 70, 68],
    [85, 88, 92, 90],
    [75, 78, 82, 80]
])

X = StandardScaler().fit_transform(data)

fa = FactorAnalysis(n_components=2, random_state=42)
factors = fa.fit_transform(X)

print("Original Data Shape:", data.shape)
print("Factor Shape:", factors.shape)
print("Factor Loadings:")
print(np.round(fa.components_, 2))
print("Factor Analysis Completed")
