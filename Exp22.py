from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.stats import loguniform

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25,
    random_state=42, stratify=y
)

# Grid Search
grid = GridSearchCV(
    SVC(),
    {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.1]
    },
    cv=3
)

grid.fit(X_train, y_train)

print("Grid Search Best Parameters:", grid.best_params_)
print("Grid Search Accuracy:",
      accuracy_score(y_test, grid.predict(X_test)))

# Random Search
random_search = RandomizedSearchCV(
    SVC(),
    {
        "C": loguniform(0.1, 10),
        "gamma": loguniform(0.001, 0.1)
    },
    n_iter=8,
    cv=3,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Random Search Best Parameters:",
      random_search.best_params_)

print("Random Search Accuracy:",
      accuracy_score(y_test, random_search.predict(X_test)))

# Bayesian Optimization
# Requires scikit-optimize: pip install scikit-optimize
try:
    from skopt import BayesSearchCV
    from skopt.space import Real

    bayes = BayesSearchCV(
        SVC(),
        {
            "C": Real(0.1, 10, prior="log-uniform"),
            "gamma": Real(0.001, 0.1, prior="log-uniform")
        },
        n_iter=10,
        cv=3,
        random_state=42
    )

    bayes.fit(X_train, y_train)

    print("Bayesian Best Parameters:", bayes.best_params_)
    print("Bayesian Accuracy:",
          accuracy_score(y_test, bayes.predict(X_test)))

except ImportError:
    print("Install scikit-optimize for Bayesian Optimization:")
    print("pip install scikit-optimize")
