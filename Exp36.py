import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    random_state=42
)

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),

    tf.keras.layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ),

    tf.keras.layers.Dense(
        16,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=10, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("L2 Regularization Loss:", loss)
print("Testing Accuracy:", accuracy)

print("Regularized Weights:")
for layer in model.layers:
    if layer.get_weights():
        print(layer.get_weights()[0])
