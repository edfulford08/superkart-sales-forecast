"""SuperKart sales forecast API (Flask).

Routes: GET / (health), GET /v1/schema, GET /v1/metadata, POST /v1/forecast, POST /v1/forecastbatch, POST /v1/drift
Security: optional API key (environment variable API_KEY), per-client rate limit (RATE_LIMIT_PER_MINUTE),
request logging, and input validation against the schema learned from the training data.
"""
import collections
import json
import logging
import os
import threading
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException

import superkart_features as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("superkart_backend")

HERE = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(HERE, "model_metadata.json")

# The metadata file (written by the notebook) names the model file, the allowed input values, and the forecast range
metadata = {}
try:
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
except (OSError, ValueError) as err:
    logging.getLogger("superkart_backend").warning("No model metadata (%s): input checks are limited to types and required fields", type(err).__name__)


def find_model_file():
    """The MODEL_FILE variable wins, then the metadata, then the newest .joblib file in this folder."""
    name = os.environ.get("MODEL_FILE") or metadata.get("model_file")
    if name:
        return name
    candidates = sorted(f for f in os.listdir(HERE) if f.endswith(".joblib"))
    return candidates[-1] if candidates else "model.joblib"


MODEL_FILE = find_model_file()
MODEL_PATH = os.path.join(HERE, MODEL_FILE)

API_KEY = os.environ.get("API_KEY", "")                               # empty means no key is required (demo mode)
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))   # 0 switches the limit off
MAX_BATCH_ROWS = int(os.environ.get("MAX_BATCH_ROWS", "20000"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ----------------------------------------------------------------------------- start-up: model and metadata
model = None
model_error = ""
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as err:  # the API still starts and reports the problem (HTTP 503)
    model_error = type(err).__name__ + ": " + str(err)
    logger.error("Could not load the model: %s", model_error)
SCHEMA = metadata.get("schema")
INTERVAL = metadata.get("interval")

# ----------------------------------------------------------------------------- rate limit, API key, logging
_hits = collections.defaultdict(collections.deque)
_lock = threading.Lock()


def client_id():
    forwarded = request.headers.get("X-Forwarded-For", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.remote_addr or "unknown")


def rate_limited(key, now=None):
    if RATE_LIMIT_PER_MINUTE <= 0:
        return False
    now = time.time() if now is None else now
    with _lock:
        window = _hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= RATE_LIMIT_PER_MINUTE:
            return True
        window.append(now)
        return False


@app.before_request
def guard():
    g.started = time.time()
    g.request_id = uuid.uuid4().hex[:8]
    if request.path == "/":
        return None                                     # the health route is always open
    if API_KEY and request.headers.get("X-API-Key", "") != API_KEY:
        return jsonify({"error": "Unauthorized: send the API key in the X-API-Key header"}), 401
    if rate_limited(client_id()):
        response = jsonify({"error": "Too many requests: the limit is " + str(RATE_LIMIT_PER_MINUTE) + " per minute"})
        response.headers["Retry-After"] = "60"
        return response, 429
    return None


@app.after_request
def log_request(response):
    elapsed = (time.time() - getattr(g, "started", time.time())) * 1000
    logger.info("req=%s %s %s -> %s in %.0f ms client=%s", getattr(g, "request_id", "-"), request.method, request.path,
                response.status_code, elapsed, client_id())
    return response


@app.errorhandler(HTTPException)
def handle_http_error(err):
    return jsonify({"error": err.name + ": " + str(err.description)}), err.code


def require_model():
    if model is None:
        raise RuntimeError("The model is not available: " + model_error)


def with_interval(predictions, frame):
    """Return the list of result dictionaries (prediction plus forecast range when available)."""
    rows = [{"Predicted_Product_Store_Sales_Total": round(float(p), 2)} for p in predictions]
    if INTERVAL:
        low, high = sf.interval_bounds(predictions, frame["Store_Type"].astype(str).tolist(), INTERVAL)
        for row, lo, hi in zip(rows, low, high):
            row["Lower_Bound"] = round(float(lo), 2)
            row["Upper_Bound"] = round(float(hi), 2)
    return rows


@app.get("/")
def home():
    healthy = model is not None
    body = {
        "service": "SuperKart Sales Forecast API",
        "status": "running" if healthy else "degraded",
        "model_loaded": healthy,
        "model_version": metadata.get("model_version"),
        "auth_required": bool(API_KEY),
        "endpoints": {"POST /v1/forecast": "single record as JSON", "POST /v1/forecastbatch": "CSV file in form field named file",
                      "POST /v1/drift": "CSV file of recent inputs in form field named file", "GET /v1/schema": "allowed values",
                      "GET /v1/metadata": "model version, metrics, forecast range"},
    }
    if not healthy:
        body["error"] = model_error
    return jsonify(body), (200 if healthy else 503)


@app.get("/v1/schema")
def schema_route():
    if not SCHEMA:
        return jsonify({"error": "The schema is not available (model_metadata.json missing)"}), 503
    return jsonify(SCHEMA)


@app.get("/v1/metadata")
def metadata_route():
    if not metadata:
        return jsonify({"error": "model_metadata.json is missing"}), 503
    safe = {k: v for k, v in metadata.items() if k not in ("drift_reference", "schema")}
    return jsonify(safe)


@app.post("/v1/forecast")
def forecast():
    try:
        require_model()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a valid JSON object")
        frame = pd.DataFrame([payload])
        features = sf.prepare_features(frame, SCHEMA)
        prediction = model.predict(features)
        result = with_interval(prediction, frame)[0]
        result["Model_Version"] = metadata.get("model_version")
        return jsonify(result)
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except RuntimeError as err:
        return jsonify({"error": str(err)}), 503
    except Exception as err:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed: " + str(err)}), 500


def read_uploaded_csv():
    uploaded = request.files.get("file")
    if uploaded is None:
        raise ValueError("Upload a CSV file in the form field named file")
    try:
        frame = pd.read_csv(uploaded)
    except Exception as err:
        raise ValueError("The uploaded file is not a valid CSV file: " + type(err).__name__) from err
    if frame.empty:
        raise ValueError("The uploaded file contains no rows")
    if len(frame) > MAX_BATCH_ROWS:
        raise ValueError("The uploaded file has " + str(len(frame)) + " rows; the limit is " + str(MAX_BATCH_ROWS))
    return frame


@app.post("/v1/forecastbatch")
def forecast_batch():
    try:
        require_model()
        batch = read_uploaded_csv()
        features = sf.prepare_features(batch, SCHEMA)
        predictions = model.predict(features)
        rows = with_interval(predictions, batch)
        output = []
        for i, row in enumerate(rows):
            record = {"row": int(i)}
            for id_col in ["Product_Id", "Store_Id"]:
                if id_col in batch.columns:
                    record[id_col] = str(batch.iloc[i][id_col])
            record.update(row)
            output.append(record)
        return jsonify({"count": len(output), "predictions": output, "model_version": metadata.get("model_version")})
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except RuntimeError as err:
        return jsonify({"error": str(err)}), 503
    except Exception as err:
        logger.exception("Batch prediction failed")
        return jsonify({"error": "Batch prediction failed: " + str(err)}), 500


@app.post("/v1/drift")
def drift():
    try:
        reference = metadata.get("drift_reference")
        if not reference:
            raise RuntimeError("The drift reference is not available (model_metadata.json missing)")
        frame = read_uploaded_csv()
        missing = [c for c in sf.REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError("Missing required fields: " + ", ".join(missing))
        return jsonify(sf.drift_report(reference, frame))
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except RuntimeError as err:
        return jsonify({"error": str(err)}), 503
    except Exception as err:
        logger.exception("Drift report failed")
        return jsonify({"error": "Drift report failed: " + str(err)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
