
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

tf.keras.utils.set_random_seed(42)

# Generate customer complaint messages
messages = [
    "payment failed",
    "money not received",
    "payment problem",
    "transaction failed",
    "cannot login",
    "login problem",
    "password incorrect",
    "account access issue",
    "delivery delayed",
    "order not delivered",
    "late delivery",
    "package missing"
]

labels = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2
])

# Convert text to sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(messages)

X = tokenizer.texts_to_sequences(messages)
X = pad_sequences(X, maxlen=5)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

# Build RNN classifier
model = Sequential([
    Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=16
    ),
    SimpleRNN(16),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=50, verbose=0)

# Predict complaint categories
predictions = model.predict(X_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

print("Classification Report")

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=["Payment", "Login", "Delivery"],
    zero_division=0
))
