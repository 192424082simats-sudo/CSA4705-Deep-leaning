import numpy as np

X = np.array([
    [0.2, 0.3],
    [0.3, 0.4],
    [0.4, 0.5],
    [0.6, 0.7],
    [0.7, 0.8],
    [0.8, 0.9],
    [0.1, 0.2],
    [0.9, 0.6]
])

y = np.array([0, 0, 0, 1, 1, 1, 0, 1])

weights = np.zeros(2)
bias = 0
learning_rate = 0.1

# Training
for epoch in range(100):

    for inputs, target in zip(X, y):

        output = np.dot(inputs, weights) + bias

        prediction = 1 if output >= 0.5 else 0

        error = target - prediction

        weights += learning_rate * error * inputs
        bias += learning_rate * error

print("Trained Weights:", weights)
print("Bias:", bias)

# Testing
test_data = [
    [0.75, 0.80],
    [0.20, 0.25]
]

for applicant in test_data:

    output = np.dot(applicant, weights) + bias

    prediction = 1 if output >= 0.5 else 0

    if prediction == 1:
        print(applicant, "-> Loan Eligible")
    else:
        print(applicant, "-> Loan Not Eligible")
