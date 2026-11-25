from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model1 = RandomForestClassifier(random_state=42)
model1.fit(X_train, y_train)
acc1 = accuracy_score(y_test, model1.predict(X_test))

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

model2 = RandomForestClassifier(random_state=42)
model2.fit(X_train_pca, y_train)
acc2 = accuracy_score(y_test, model2.predict(X_test_pca))

print("Accuracy Before PCA:", round(acc1 * 100, 2), "%")
print("Accuracy After PCA:", round(acc2 * 100, 2), "%")
