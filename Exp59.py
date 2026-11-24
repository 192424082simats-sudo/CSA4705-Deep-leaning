
import numpy as np

# Initial gradient
gradient = 1.0

# Gradient multiplier
weight = 0.5

print("Vanishing Gradient Demonstration")
print("Time Step | Gradient Value")

for step in range(1, 11):
    gradient = gradient * weight
    print(step, "       |", round(gradient, 6))

print("\nFinal Gradient:", round(gradient, 6))

if gradient < 0.01:
    print("Vanishing Gradient Problem Detected")
