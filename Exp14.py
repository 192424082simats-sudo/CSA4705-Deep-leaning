w = 0
learning_rate = 0.1

for i in range(50):
    gradient = 2 * (w - 5)
    w = w - learning_rate * gradient

cost = (w - 5) ** 2 + 3

print("Optimized Weight:", w)
print("Minimum Cost:", cost)
