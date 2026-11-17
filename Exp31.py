import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

X = np.array([
    [1000, 2],
    [1200, 2],
    [1500, 3],
    [1800, 3],
    [2000, 4],
    [2300, 4]
])

y = np.array([30, 36, 48, 58, 68, 78])

model = MLPRegressor(
    hidden_layer_sizes=(10,),
    max_iter=5000,
    random_state=42
)

model.fit(X, y)

predictions = model.predict(X)

print("Actual Prices:", y)
print("Predicted Prices:", predictions)

print("Mean Squared Error:",
      mean_squared_error(y, predictions))
