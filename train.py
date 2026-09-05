import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense


# Load dataset
data = pd.read_csv("dataset.csv")

print("Total messages:", len(data))
print(data["label"].value_counts())


# Convert labels
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# Separate messages and labels
messages = data["message"].astype(str).values
labels = data["label"].values


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


# Create tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)


# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


# Make all sequences the same length
max_length = 50

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=max_length,
    padding="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=max_length,
    padding="post"
)


# Build LSTM model
model = Sequential([
    Embedding(
        input_dim=5000,
        output_dim=32
    ),

    LSTM(32),

    Dropout(0.3),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")
])


# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Train model
model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)


# Evaluate model
loss, accuracy = model.evaluate(
    X_test_pad,
    y_test,
    verbose=0
)

print("Test Accuracy:", accuracy)


# Save model
model.save("spam_model.keras")

print("Model saved as spam_model.keras")


# Save tokenizer
tokenizer_json = tokenizer.to_json()

with open("tokenizer.json", "w") as file:
    file.write(tokenizer_json)

print("Tokenizer saved as tokenizer.json")