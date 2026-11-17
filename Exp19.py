from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

X, y = make_regression(
    n_samples=150,
    n_features=6,
    noise=20,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

models = {
    "L1 Regularization": Lasso(alpha=1.0),
    "L2 Regularization": Ridge(alpha=1.0)
}

for name, algorithm in models.items():

    model = make_pipeline(
        StandardScaler(),
        algorithm
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    print(name)
    print("Coefficients:", model[-1].coef_)
    print("MSE:", mean_squared_error(y_test, prediction))
