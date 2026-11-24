import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1,28,28,1) / 255.0
X_test = X_test.reshape(-1,28,28,1) / 255.0

filters = [16, 32, 64]
filter_sizes = [(3,3), (5,5)]

for f in filters:
    for size in filter_sizes:

        model = Sequential([
            Conv2D(f, size, activation='relu',
                   input_shape=(28,28,1)),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(X_train, y_train,
                  epochs=2, batch_size=64, verbose=0)

        loss, accuracy = model.evaluate(
            X_test, y_test, verbose=0
        )

        print("Filters:", f)
        print("Filter Size:", size)
        print("Accuracy:", accuracy)
        print("--------------------")
