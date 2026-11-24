
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

tf.keras.utils.set_random_seed(42)

reviews = [
    "This product is excellent",
    "I love this product",
    "Very good quality",
    "Amazing product",
    "This is wonderful",
    "The product is fantastic",
    "I am happy with this",
    "Excellent performance",
    "This product is bad",
    "I hate this product",
    "Very poor quality",
    "Worst product",
    "This is terrible",
    "Not worth buying",
    "I am disappointed",
    "Bad performance"
]

labels = np.array([1]*8 + [0]*8)

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(reviews)

X = tokenizer.texts_to_sequences(reviews)
X = pad_sequences(X, maxlen=6)

model = Sequential([
    Embedding(100, 16, input_length=6),
    SimpleRNN(16),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, labels, epochs=30, verbose=0)

test_reviews = [
    "excellent product",
    "terrible product"
]

test = tokenizer.texts_to_sequences(test_reviews)
test = pad_sequences(test, maxlen=6)

predictions = model.predict(test, verbose=0)

for review, prediction in zip(test_reviews, predictions):
    sentiment = "Positive" if prediction[0] >= 0.5 else "Negative"
    print(review, ":", sentiment)
