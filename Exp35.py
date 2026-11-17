import tensorflow as tf

(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

X_train = X_train / 255.0
X_test = X_test / 255.0

# Model without dropout
model1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model1.fit(X_train, y_train, epochs=5, verbose=0)

loss1, accuracy1 = model1.evaluate(
    X_test, y_test, verbose=0
)

# Model with dropout
model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model2.fit(X_train, y_train, epochs=5, verbose=0)

loss2, accuracy2 = model2.evaluate(
    X_test, y_test, verbose=0
)

print("Without Dropout Accuracy:", accuracy1)
print("With Dropout Accuracy:", accuracy2)
print("Without Dropout Loss:", loss1)
print("With Dropout Loss:", loss2)
