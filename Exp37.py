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

# Model without regularization
model1 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model1.fit(X_train, y_train, epochs=15, verbose=0)

train_loss1, train_acc1 = model1.evaluate(
    X_train, y_train, verbose=0
)

test_loss1, test_acc1 = model1.evaluate(
    X_test, y_test, verbose=0
)

# Model with L2 regularization
model2 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ),
    tf.keras.layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model2.fit(X_train, y_train, epochs=15, verbose=0)

train_loss2, train_acc2 = model2.evaluate(
    X_train, y_train, verbose=0
)

test_loss2, test_acc2 = model2.evaluate(
    X_test, y_test, verbose=0
)

print("WITHOUT REGULARIZATION")
print("Training Accuracy:", train_acc1)
print("Testing Accuracy:", test_acc1)

print("\nWITH L2 REGULARIZATION")
print("Training Accuracy:", train_acc2)
print("Testing Accuracy:", test_acc2)
