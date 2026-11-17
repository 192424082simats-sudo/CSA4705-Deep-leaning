import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score

X, y = make_classification(
    n_samples=500,
    n_features=8,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]
predictions = model.predict(X_test)

print("Binary Cross Entropy Loss:",
      log_loss(y_test, probabilities))

print("Accuracy:",
      accuracy_score(y_test, predictions))

print("Predictions:", predictions)
