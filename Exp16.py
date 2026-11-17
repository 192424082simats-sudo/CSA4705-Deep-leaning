from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(
    n_samples=300,
    noise=0.25,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

print("Training Accuracy:",
      accuracy_score(y_train, train_prediction))

print("Testing Accuracy:",
      accuracy_score(y_test, test_prediction))

print("Overfitting occurs when training accuracy is high but testing accuracy is low.")
