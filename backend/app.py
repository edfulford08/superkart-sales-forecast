import logging
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("superkart_backend")

# Location of the serialized model next to this file
MODEL_FILE = "superkart_sales_prediction_model_v1_0.joblib"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILE)

# Same reference year used when the model was trained (Store_Age = REFERENCE_YEAR - Store_Establishment_Year)
REFERENCE_YEAR = 2026

NUMERIC_FEATURES = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age"]
CATEGORICAL_FEATURES = ["Product_Sugar_Content", "Product_Type", "Store_Size",
                        "Store_Location_City_Type", "Store_Type", "Product_Id_Prefix"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Raw columns a caller must provide
REQUIRED_COLUMNS = ["Product_Id", "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
                    "Product_Type", "Product_MRP", "Store_Establishment_Year", "Store_Size",
                    "Store_Location_City_Type", "Store_Type"]
NUMERIC_INPUTS = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]

app = Flask(__name__)

# Load the model once at start-up. If loading fails the API still starts and reports the problem
# (HTTP 503) instead of crashing, so the cause is visible in the logs and on the health route.
model = None
model_error = ""
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as err:
    model_error = type(err).__name__ + ": " + str(err)
    logger.error("Could not load the model: %s", model_error)


def prepare_features(df):
    """Validate raw records and apply the same cleaning and feature engineering used in training."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    df = df.copy()
    for col in NUMERIC_INPUTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[NUMERIC_INPUTS].isnull().any().any():
        raise ValueError("Numeric fields must contain valid numbers")
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].astype(str).str.strip().replace({"reg": "Regular"})
    df["Product_Id_Prefix"] = df["Product_Id"].astype(str).str[:2]
    df["Store_Age"] = REFERENCE_YEAR - df["Store_Establishment_Year"]
    return df[MODEL_FEATURES]


def require_model():
    if model is None:
        raise RuntimeError("The model is not available: " + model_error)


@app.errorhandler(HTTPException)
def handle_http_error(err):
    # Unknown routes, wrong methods, and similar problems return JSON instead of an HTML page
    return jsonify({"error": err.name + ": " + str(err.description)}), err.code


@app.get("/")
def home():
    healthy = model is not None
    body = {
        "service": "SuperKart Sales Forecast API",
        "status": "running" if healthy else "degraded",
        "model_loaded": healthy,
        "endpoints": {"POST /v1/forecast": "single record as JSON",
                      "POST /v1/forecastbatch": "CSV file in form field named file"},
    }
    if not healthy:
        body["error"] = model_error
    return jsonify(body), (200 if healthy else 503)


@app.post("/v1/forecast")
def forecast():
    try:
        require_model()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a valid JSON object")
        features = prepare_features(pd.DataFrame([payload]))
        prediction = float(model.predict(features)[0])
        return jsonify({"Predicted_Product_Store_Sales_Total": round(prediction, 2)})
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except RuntimeError as err:
        return jsonify({"error": str(err)}), 503
    except Exception as err:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed: " + str(err)}), 500


@app.post("/v1/forecastbatch")
def forecast_batch():
    try:
        require_model()
        uploaded = request.files.get("file")
        if uploaded is None:
            raise ValueError("Upload a CSV file in the form field named file")
        batch = pd.read_csv(uploaded)
        if batch.empty:
            raise ValueError("The uploaded file contains no rows")
        features = prepare_features(batch)
        predictions = model.predict(features)
        output = []
        for i, value in enumerate(predictions):
            row = {"row": int(i)}
            for id_col in ["Product_Id", "Store_Id"]:
                if id_col in batch.columns:
                    row[id_col] = str(batch.iloc[i][id_col])
            row["Predicted_Product_Store_Sales_Total"] = round(float(value), 2)
            output.append(row)
        return jsonify({"count": len(output), "predictions": output})
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except RuntimeError as err:
        return jsonify({"error": str(err)}), 503
    except Exception as err:
        logger.exception("Batch prediction failed")
        return jsonify({"error": "Batch prediction failed: " + str(err)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
