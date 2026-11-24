
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)

# Word embeddings
words = {
    "good": np.array([1.0, 0.5]),
    "great": np.array([1.0, 0.8]),
    "excellent": np.array([1.0, 0.9]),
    "bad": np.array([-1.0, -0.5]),
    "poor": np.array([-1.0, -0.8]),
    "terrible": np.array([-1.0, -0.9])
}

# Recursive composition weights
W = np.random.randn(2, 4) * 0.2
b = np.zeros(2)

# Recursive neural network function
def recursive_network(tree):

    if isinstance(tree, str):
        return words[tree]

    left = recursive_network(tree[0])
    right = recursive_network(tree[1])

    combined = np.concatenate([left, right])

    output = np.tanh(np.dot(W, combined) + b)

    return output

# Training sentence trees
trees = [
    ("good", "great"),
    ("great", "excellent"),
    ("excellent", "good"),
    ("bad", "poor"),
    ("poor", "terrible"),
    ("terrible", "bad"),
    (("good", "great"), "excellent"),
    (("bad", "poor"), "terrible")
]

labels = np.array([1, 1, 1, 0, 0, 0, 1, 0])

# Create recursive feature vectors
features = np.array([
    recursive_network(tree) for tree in trees
])

# Train sentiment classifier
classifier = LogisticRegression(random_state=42)
classifier.fit(features, labels)

# Predict sentiment
predictions = classifier.predict(features)

accuracy = accuracy_score(labels, predictions)

print("Recursive Neural Network")
print("Sentiment Classification Results")

for i, prediction in enumerate(predictions):
    sentiment = "Positive" if prediction == 1 else "Negative"
    print("Sentence", i+1, ":", sentiment)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Recursive Classification Completed")
