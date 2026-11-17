import tensorflow as tf
import numpy as np

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Predict digit
prediction = model.predict(X_test[:1])
print("Predicted Digit:", np.argmax(prediction))
print("Actual Digit:", y_test[0])
