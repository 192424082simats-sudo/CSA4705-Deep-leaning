import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1,28,28,1) / 255.0
X_test = X_test.reshape(-1,28,28,1) / 255.0

def create_model(use_bn):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu',
                     input_shape=(28,28,1)))

    if use_bn:
        model.add(BatchNormalization())

    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

for bn in [False, True]:

    model = create_model(bn)

    history = model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )

    loss, accuracy = model.evaluate(
        X_test, y_test, verbose=0
    )

    print("Batch Normalization:", bn)
    print("Test Accuracy:", accuracy)
    print("Final Training Loss:",
          history.history['loss'][-1])
