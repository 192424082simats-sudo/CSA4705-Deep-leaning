import numpy as np
from sklearn.decomposition import FastICA

np.random.seed(42)

time = np.linspace(0, 8, 1000)

s1 = np.sin(2 * time)
s2 = np.sign(np.sin(3 * time))

S = np.c_[s1, s2]
S = S / S.std(axis=0)

A = np.array([
    [1, 0.5],
    [0.5, 1]
])

X = S @ A.T

ica = FastICA(n_components=2, random_state=42)
S_ = ica.fit_transform(X)

print("Original Mixed Data Shape:", X.shape)
print("Separated Data Shape:", S_.shape)
print("ICA Source Separation Completed")
