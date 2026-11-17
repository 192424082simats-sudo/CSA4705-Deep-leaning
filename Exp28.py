
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = make_classification(
    n_samples=500,
    n_features=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for activation in ['relu', 'logistic', 'tanh']:

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(20,),
            activation=activation,
            max_iter=3000,
            learning_rate_init=0.001,
            random_state=42
        )
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    print("\nActivation:", activation)
    print("Accuracy:", accuracy_score(y_test, prediction))

