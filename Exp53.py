
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

tf.keras.utils.set_random_seed(42)

sentences = [
    "deep learning is very useful",
    "machine learning is very powerful",
    "deep learning uses neural networks",
    "machine learning uses algorithms",
    "neural networks learn from data",
    "deep learning is interesting"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

X, y = [], []

for sentence in sentences:
    tokens = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(tokens)):
        X.append(tokens[:i])
        y.append(tokens[i])

max_length = max(len(x) for x in X)
X = pad_sequences(X, maxlen=max_length)

y = np.array(y)

vocab_size = len(tokenizer.word_index) + 1

model = Sequential([
    Embedding(vocab_size, 16),
    SimpleRNN(32),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=100, verbose=0)

input_text = "deep learning"
sequence = tokenizer.texts_to_sequences([input_text])
sequence = pad_sequences(sequence, maxlen=max_length)

prediction = model.predict(sequence, verbose=0)
next_word_id = np.argmax(prediction[0])

word = tokenizer.index_word.get(next_word_id, "Unknown")

print("Input Text:", input_text)
print("Predicted Next Word:", word)
