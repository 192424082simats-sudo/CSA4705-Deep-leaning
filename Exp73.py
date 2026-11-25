import numpy as np
from sklearn.decomposition import FastICA

np.random.seed(42)

t = np.linspace(0, 10, 1000)

voice1 = np.sin(2 * np.pi * 5 * t)
voice2 = np.sin(2 * np.pi * 10 * t)

S = np.c_[voice1, voice2]

A = np.array([
    [1, 0.5],
    [0.5, 1]
])

mixed = S @ A.T

ica = FastICA(n_components=2, random_state=42)
separated = ica.fit_transform(mixed)

print("Mixed Signals Shape:", mixed.shape)
print("Separated Signals Shape:", separated.shape)
print("Number of Sources:", separated.shape[1])
print("Blind Source Separation Completed")
