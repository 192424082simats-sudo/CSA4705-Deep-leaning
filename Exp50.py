
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

tf.keras.utils.set_random_seed(42)

text = "deep learning is an important technology "

chars = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}

seq_length = 5
X, y = [], []

for i in range(len(text) - seq_length):
    X.append([char_to_index[c] for c in text[i:i+seq_length]])
    y.append(char_to_index[text[i+seq_length]])

X = np.array(X)
y = np.array(y)

X = tf.keras.utils.to_categorical(X, num_classes=len(chars))
y = tf.keras.utils.to_categorical(y, num_classes=len(chars))

model = Sequential([
    tf.keras.Input(shape=(seq_length, len(chars))),
    SimpleRNN(64),
    Dense(len(chars), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy"
)

model.fit(X, y, epochs=100, verbose=0)

seed = "learn"
generated = seed

for _ in range(30):
    encoded = [
        char_to_index[c] for c in generated[-seq_length:]
    ]

    encoded = tf.keras.utils.to_categorical(
        [encoded], num_classes=len(chars)
    )

    prediction = model.predict(encoded, verbose=0)
    next_char = index_to_char[int(np.argmax(prediction[0]))]
    generated += next_char

print("Generated Text:")
print(generated)
