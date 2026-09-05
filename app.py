from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained LSTM model
model = tf.keras.models.load_model("spam_model.keras")

# Load the tokenizer
with open("tokenizer.json", "r") as file:
    tokenizer = tokenizer_from_json(file.read())


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None

    if request.method == "POST":

        message = request.form["message"]

        # Convert message into numbers
        sequence = tokenizer.texts_to_sequences([message])

        # Make the sequence length equal to training length
        padded = pad_sequences(
            sequence,
            maxlen=50,
            padding="post"
        )

        # Get prediction from LSTM
        prediction = model.predict(
            padded,
            verbose=0
        )[0][0]

        # 1 = spam
        # 0 = ham
        if prediction >= 0.5:

            result = "🚨 SPAM"
            confidence = prediction * 100

        else:

            result = "✅ NOT SPAM"
            confidence = (1 - prediction) * 100

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)