import numpy as np
import matplotlib.pyplot as plt

samples = np.random.normal(25, 4, 1000)

print("First 10 Samples:", samples[:10])
print("Mean:", np.mean(samples))
print("Standard Deviation:", np.std(samples))

plt.hist(samples, bins=25, density=True)
plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.show()
