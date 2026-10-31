
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from flask_cors import CORS

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "model.pkl"

app = Flask(__name__)

CORS(app, resources={
    r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}
})

metrics = PrometheusMetrics(app)

model = joblib.load(MODEL_PATH)

FEATURES = [
    "current_occupancy_beds",
    "previous_occupancy_beds",
    "two_weeks_ago_occupancy_beds",
    "total_beds_7_day_avg",
    "current_occupancy_rate",
]


@app.get("/")
def home():
    return jsonify({
        "message": "Hospital Bed Occupancy Prediction API is running",
        "endpoint": "POST /predict",
        "features_required": FEATURES,
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object"}), 400

    missing = [feature for feature in FEATURES if feature not in data]
    if missing:
        return jsonify({
            "error": "Missing required input fields",
            "missing_fields": missing,
        }), 400

    try:
        values = [float(data[feature]) for feature in FEATURES]
    except (TypeError, ValueError):
        return jsonify({
            "error": "All feature values must be numeric"
        }), 400

    if not all(np.isfinite(value) for value in values):
        return jsonify({
            "error": "Feature values must be finite numbers"
        }), 400

    current, previous, two_weeks_ago, capacity, occupancy_rate = values

    if min(current, previous, two_weeks_ago) < 0:
        return jsonify({
            "error": "Occupancy values cannot be negative"
        }), 400

    if capacity <= 0:
        return jsonify({
            "error": "Total bed capacity must be greater than zero"
        }), 400

    if not 0 <= occupancy_rate <= 1.5:
        return jsonify({
            "error": "Occupancy rate must be between 0 and 1.5"
        }), 400

    # Keep the same feature names and order used during model training.
    import pandas as pd

    features = pd.DataFrame([{
        "current_occupancy_beds": current,
        "previous_occupancy_beds": previous,
        "two_weeks_ago_occupancy_beds": two_weeks_ago,
        "total_beds_7_day_avg": capacity,
        "current_occupancy_rate": occupancy_rate,
    }])

    prediction = float(model.predict(features)[0])
    prediction = max(0.0, prediction)
    prediction = round(prediction)

    return jsonify({
        "predicted_occupancy_beds_next_week": round(prediction, 2),
        "current_occupancy_beds": current,
        "model": "Random Forest",
        "unit": "beds",
        "forecast_horizon": "7 days",
        "warning": (
            "Experimental prediction based on historical US hospital data; "
            "not a substitute for operational clinical decisions."
        ),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

