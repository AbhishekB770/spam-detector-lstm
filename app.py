from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load lightweight spam detection model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        message = request.form["message"]

        # Convert message into TF-IDF features
        message_vector = vectorizer.transform([message])

        # Predict spam/ham
        prediction = model.predict(message_vector)[0]
        probabilities = model.predict_proba(message_vector)[0]

        if prediction == 1:
            result = "SPAM"
            confidence = probabilities[1] * 100
        else:
            result = "NOT SPAM"
            confidence = probabilities[0] * 100

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )


if __name__ == "__main__":
    import os

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )