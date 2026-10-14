"""Flask API: loads the saved model and serves the React frontend.
Run: python app.py  ->  http://localhost:5000"""
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from features import RAW, build_features

app = Flask(__name__, static_folder="frontend", static_url_path="")
bundle = joblib.load("models/best_model.joblib")
model = bundle["model"]
with open("models/metrics.json") as f:
    info = json.load(f)

@app.after_request
def cors(resp):  # lets the frontend call the API even if served from another port (e.g. VS Code Live Server)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.get("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.get("/health")
def health():
    return jsonify(status="ok", model=bundle["name"])

@app.get("/model-info")
def model_info():
    return jsonify(info)

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)
    data = request.get_json(silent=True) or {}
    missing = [k for k in RAW if k not in data]
    if missing:
        return jsonify(error=f"Missing fields: {', '.join(missing)}"), 400
    try:
        row = {k: float(data[k]) for k in RAW}
    except (TypeError, ValueError):
        return jsonify(error="All fields must be numbers (use 1/0 for yes/no)."), 400
    if any(row[k] < 0 for k in ("followers", "following", "posts", "account_age_days")):
        return jsonify(error="Counts and account age cannot be negative."), 400
    proba = float(model.predict_proba(build_features(pd.DataFrame([row])))[0, 1])
    return jsonify(
        prediction="Likely Fake Account" if proba >= 0.5 else "Real Account",
        is_fake=proba >= 0.5,
        probability_fake=round(proba, 4),
        model=bundle["name"],
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
