import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("dataset.csv")

# Convert labels
data["label"] = data["label"].map({"ham": 0, "spam": 1})

messages = data["message"].astype(str)
labels = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)

# Train lightweight classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Lightweight model created successfully!")
print("Model saved as spam_model.pkl")
print("Vectorizer saved as vectorizer.pkl")