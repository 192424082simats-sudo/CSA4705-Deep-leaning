import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = np.linspace(-3, 3, 80).reshape(-1, 1)

y = 0.5 * X[:, 0]**2 + X[:, 0] + 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

for degree in [1, 2, 10]:

    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    train_error = mean_squared_error(
        y_train, model.predict(X_train)
    )

    test_error = mean_squared_error(
        y_test, model.predict(X_test)
    )

    print("Degree:", degree)
    print("Training Error:", train_error)
    print("Testing Error:", test_error)
