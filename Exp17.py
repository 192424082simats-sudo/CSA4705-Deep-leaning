import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([
    [1000, 2, 1],
    [1200, 2, 1],
    [1500, 3, 2],
    [1800, 3, 2],
    [2000, 4, 3],
    [2300, 4, 3]
])

y = np.array([30, 36, 48, 58, 68, 78])

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions:", predictions)

print("Mean Squared Error:",
      mean_squared_error(y, predictions))

print("New House Price:",
      model.predict([[1400, 3, 2]]))
