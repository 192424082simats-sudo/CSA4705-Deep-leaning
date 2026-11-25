import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

X = np.array([
    [80, 70, 90, 85],
    [60, 65, 70, 75],
    [90, 85, 95, 90],
    [50, 55, 60, 65],
    [75, 80, 85, 80],
    [65, 60, 75, 70]
])

X_scaled = StandardScaler().fit_transform(X)

fa = FactorAnalysis(n_components=2, random_state=42)
factors = fa.fit_transform(X_scaled)

print("Original Shape:", X.shape)
print("Reduced Shape:", factors.shape)
print("Customer Behaviour Factors:")
print(np.round(factors, 2))
print("Factor Analysis Completed")
