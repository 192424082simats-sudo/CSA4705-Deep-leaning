import numpy as np

np.random.seed(42)

mean = [10, 20, 30]

covariance = [
    [1, 0.5, 0.2],
    [0.5, 2, 0.3],
    [0.2, 0.3, 1]
]

data = np.random.multivariate_normal(
    mean, covariance, 1000
)

estimated_mean = np.mean(data, axis=0)
estimated_covariance = np.cov(data, rowvar=False)

print("Generated Data Shape:", data.shape)
print("Estimated Mean:")
print(np.round(estimated_mean, 2))
print("Estimated Covariance:")
print(np.round(estimated_covariance, 2))
print("Gaussian Data Generation Completed")
