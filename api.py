from flask import Flask, request, jsonify
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text", "")
    vect = vectorizer.transform([text])
    pred = model.predict(vect)[0]
    label = "REAL" if pred == 1 else "FAKE"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
