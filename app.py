from flask import Flask, render_template, request
import pickle
import csv
import os

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

CSV_FILE = "user_reviews.csv"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""

    if request.method == "POST":
        review = request.form.get("review")

        if review:
            # Transform input
            review_vector = vectorizer.transform([review])
            prediction = model.predict(review_vector)[0]

            # Convert to readable text
            if prediction == "positive":
                prediction_text = "Positive Review 😊"
            else:
                prediction_text = "Negative Review 😞"

            # Save review + sentiment
            save_review(review, prediction)

    return render_template("index.html", prediction=prediction_text)


def save_review(review, sentiment):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header only once
        if not file_exists:
            writer.writerow(["review", "sentiment"])

        writer.writerow([review, sentiment])


if __name__ == "__main__":
    app.run(debug=True)
