import numpy as np

a = np.float32(0.1) + np.float32(0.2)
b = np.float64(0.1) + np.float64(0.2)

print("Float32 Result:", a)
print("Float64 Result:", b)

print("Float64 equals 0.3:", b == 0.3)
print("Using tolerance:", np.isclose(b, 0.3))

large = np.float32(1e8)

print("Precision Test:",
      (large + np.float32(1)) - large)
