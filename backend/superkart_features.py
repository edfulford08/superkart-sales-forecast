"""Shared feature engineering, input validation, forecast intervals and drift monitoring for SuperKart.

This file is written by the notebook and imported by the notebook, the Flask API, and the tests, so the training
logic and the serving logic cannot drift apart.
"""
import re

import numpy as np
import pandas as pd

# Year used to turn Store_Establishment_Year into Store_Age (set once in the notebook configuration cell)
REFERENCE_YEAR = 2026

NUMERIC_FEATURES = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age"]
CATEGORICAL_FEATURES = ["Product_Sugar_Content", "Product_Type", "Store_Size",
                        "Store_Location_City_Type", "Store_Type", "Product_Id_Prefix"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Raw columns a caller must provide (Store_Id and the target are not needed)
REQUIRED_COLUMNS = ["Product_Id", "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
                    "Product_Type", "Product_MRP", "Store_Establishment_Year", "Store_Size",
                    "Store_Location_City_Type", "Store_Type"]
NUMERIC_INPUTS = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
CATEGORICAL_INPUTS = ["Product_Sugar_Content", "Product_Type", "Store_Size", "Store_Location_City_Type", "Store_Type"]
DROP_COLUMNS = ["Product_Id", "Store_Id", "Store_Establishment_Year"]
SUGAR_FIXES = {"reg": "Regular"}
PRODUCT_ID_PATTERN = re.compile(r"^[A-Za-z]{2}\d+$")

# Columns compared when checking for drift in the inputs
DRIFT_NUMERIC = ["Product_Weight", "Product_Allocated_Area", "Product_MRP"]
DRIFT_CATEGORICAL = ["Product_Sugar_Content", "Product_Type", "Store_Size", "Store_Location_City_Type",
                     "Store_Type", "Product_Id_Prefix"]
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


# ----------------------------------------------------------------------------- feature engineering
def engineer_features(df):
    """Clean the labels and build the model features; the target column is kept when present."""
    df = df.copy()
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].astype(str).str.strip().replace(SUGAR_FIXES)
    df["Product_Id_Prefix"] = df["Product_Id"].astype(str).str[:2]
    df["Store_Age"] = REFERENCE_YEAR - df["Store_Establishment_Year"]
    return df.drop(columns=DROP_COLUMNS, errors="ignore")


def add_extra_features(df, store_types=()):
    """Experimental features used only for the feature tests in the notebook (not by the deployed model)."""
    df = df.copy()
    df["MRP_per_Weight"] = df["Product_MRP"] / df["Product_Weight"].clip(lower=1e-6)
    df["MRP_x_Area"] = df["Product_MRP"] * df["Product_Allocated_Area"]
    for store_type in store_types:
        df["MRP_x_" + str(store_type).replace(" ", "_")] = df["Product_MRP"] * (df["Store_Type"] == store_type).astype(float)
    return df


# ----------------------------------------------------------------------------- schema and validation
def build_schema(raw_df):
    """Describe the allowed values of every input, learned from the training data."""
    clean = raw_df.copy()
    clean["Product_Sugar_Content"] = clean["Product_Sugar_Content"].astype(str).str.strip().replace(SUGAR_FIXES)
    schema = {
        "categories": {c: sorted(clean[c].astype(str).unique().tolist()) for c in CATEGORICAL_INPUTS},
        "product_id_prefixes": sorted(clean["Product_Id"].astype(str).str[:2].unique().tolist()),
        "numeric": {},
        "defaults": {},
    }
    for col in NUMERIC_INPUTS:
        low, high = float(clean[col].min()), float(clean[col].max())
        span = high - low
        if col == "Store_Establishment_Year":
            allowed_min, allowed_max = 1950.0, float(REFERENCE_YEAR)
        else:
            allowed_min, allowed_max = max(0.0, low - 0.5 * span), high + 0.5 * span
            if col == "Product_Allocated_Area":
                allowed_max = min(allowed_max, 1.0)
        schema["numeric"][col] = {"observed_min": low, "observed_max": high, "min": allowed_min, "max": allowed_max}
        schema["defaults"][col] = float(clean[col].median())
    for col in CATEGORICAL_INPUTS:
        schema["defaults"][col] = str(clean[col].astype(str).mode().iloc[0])
    schema["defaults"]["Product_Id"] = str(clean["Product_Id"].astype(str).iloc[0])
    return schema


def validate_records(df, schema=None, max_problems=20):
    """Return a list of readable problems with the raw records (an empty list means the records are valid)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return ["Missing required fields: " + ", ".join(missing)]
    problems = []
    numeric = {}
    for col in NUMERIC_INPUTS:
        values = pd.to_numeric(df[col], errors="coerce")
        numeric[col] = values
        bad = ~np.isfinite(values.to_numpy(dtype=float))
        for row in np.flatnonzero(bad)[:max_problems]:
            problems.append(f"Row {row + 1}: {col} must be a valid number")
    if schema:
        for col, rule in schema.get("numeric", {}).items():
            if col in numeric:
                values = numeric[col].to_numpy(dtype=float)
                out = np.isfinite(values) & ((values < rule["min"]) | (values > rule["max"]))
                for row in np.flatnonzero(out)[:max_problems]:
                    problems.append(f"Row {row + 1}: {col} = {values[row]:g} is outside the allowed range {rule['min']:g} to {rule['max']:g}")
        sugar = df["Product_Sugar_Content"].astype(str).str.strip().replace(SUGAR_FIXES)
        for col, allowed in schema.get("categories", {}).items():
            series = sugar if col == "Product_Sugar_Content" else df[col].astype(str)
            bad = ~series.isin(allowed)
            for row in np.flatnonzero(bad.to_numpy())[:max_problems]:
                problems.append(f"Row {row + 1}: {col} = '{series.iloc[row]}' is not one of {allowed}")
        ids = df["Product_Id"].astype(str)
        for row in np.flatnonzero(~ids.str.match(PRODUCT_ID_PATTERN).to_numpy())[:max_problems]:
            problems.append(f"Row {row + 1}: Product_Id '{ids.iloc[row]}' must be two letters followed by digits")
        prefixes = schema.get("product_id_prefixes")
        if prefixes:
            ok_pattern = ids.str.match(PRODUCT_ID_PATTERN).to_numpy()
            bad = ok_pattern & ~ids.str[:2].isin(prefixes).to_numpy()
            for row in np.flatnonzero(bad)[:max_problems]:
                problems.append(f"Row {row + 1}: Product_Id prefix '{ids.iloc[row][:2]}' is not one of {prefixes}")
    if len(problems) > max_problems:
        extra = len(problems) - max_problems
        problems = problems[:max_problems] + [f"...and {extra} more problems"]
    return problems


def prepare_features(df, schema=None):
    """Validate raw records and return the model features, or raise ValueError with every problem found."""
    problems = validate_records(df, schema)
    if problems:
        raise ValueError("; ".join(problems))
    features = engineer_features(df)
    for col in NUMERIC_FEATURES:
        features[col] = pd.to_numeric(features[col])
    return features[MODEL_FEATURES]


# ----------------------------------------------------------------------------- forecast intervals
def interval_bounds(predictions, store_types, interval):
    """Lower and upper bound of the forecast range (store type specific half widths, never below zero)."""
    predictions = np.asarray(predictions, dtype=float)
    widths = interval.get("half_width", {})
    fallback = interval["global_half_width"]
    half = np.array([widths.get(str(s), fallback) for s in store_types], dtype=float)
    return np.maximum(predictions - half, 0.0), predictions + half


def conformal_half_widths(store_types, abs_residuals, coverage):
    """Split conformal half width per store type (Mondrian) and overall, from out-of-fold absolute residuals."""
    store_types = np.asarray(store_types)
    abs_residuals = np.asarray(abs_residuals, dtype=float)

    def quantile(values):
        n = len(values)
        level = min(1.0, np.ceil((n + 1) * coverage) / n)
        return float(np.quantile(values, level, method="higher"))

    return {
        "global_half_width": quantile(abs_residuals),
        "half_width": {str(s): quantile(abs_residuals[store_types == s]) for s in np.unique(store_types)},
    }


# ----------------------------------------------------------------------------- drift and monitoring
def _psi(expected, actual, eps=1e-4):
    expected = np.clip(np.asarray(expected, dtype=float), eps, None)
    actual = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _status(value):
    if value >= PSI_SIGNIFICANT:
        return "significant"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def _with_drift_columns(raw_df):
    df = raw_df.copy()
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].astype(str).str.strip().replace(SUGAR_FIXES)
    df["Product_Id_Prefix"] = df["Product_Id"].astype(str).str[:2]
    return df


def build_drift_reference(raw_df, n_bins=10):
    """Bin edges and shares of the training inputs, stored with the model and used to detect drift later."""
    df = _with_drift_columns(raw_df)
    reference = {"n_rows": int(len(df)), "numeric": {}, "categorical": {}}
    for col in DRIFT_NUMERIC:
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        inner = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
        counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
        reference["numeric"][col] = {"edges": inner.tolist(), "shares": (counts / counts.sum()).tolist()}
    for col in DRIFT_CATEGORICAL:
        shares = df[col].astype(str).value_counts(normalize=True)
        reference["categorical"][col] = {k: float(v) for k, v in shares.items()}
    return reference


def drift_report(reference, raw_df):
    """Population stability index (PSI) of every monitored input against the training data."""
    df = _with_drift_columns(raw_df)
    rows = []
    for col, ref in reference["numeric"].items():
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        counts = np.bincount(np.searchsorted(np.asarray(ref["edges"]), values, side="right"), minlength=len(ref["shares"]))
        value = _psi(ref["shares"], counts / counts.sum())
        rows.append({"feature": col, "type": "numeric", "psi": round(value, 4), "status": _status(value)})
    for col, ref in reference["categorical"].items():
        observed = df[col].astype(str).value_counts(normalize=True)
        categories = sorted(set(ref) | set(observed.index))
        expected = [ref.get(c, 0.0) for c in categories]
        actual = [float(observed.get(c, 0.0)) for c in categories]
        value = _psi(expected, actual)
        rows.append({"feature": col, "type": "categorical", "psi": round(value, 4), "status": _status(value)})
    order = {"stable": 0, "moderate": 1, "significant": 2}
    overall = max((r["status"] for r in rows), key=lambda s: order[s]) if rows else "stable"
    return {"rows": int(len(df)), "overall_status": overall, "thresholds": {"moderate": PSI_MODERATE, "significant": PSI_SIGNIFICANT},
            "features": rows}


def error_monitor(actual, predicted, baseline_mae):
    """Compare recent forecast errors with the error measured when the model was trained."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(np.mean(np.abs(actual - predicted)))
    aggregate = float((predicted.sum() - actual.sum()) / actual.sum() * 100)
    return {"mae": mae, "mae_ratio": mae / baseline_mae, "aggregate_error_pct": aggregate}


def needs_retraining(drift_status, mae_ratio, age_days, policy):
    """Apply the retraining policy; returns (decision, reasons)."""
    reasons = []
    if drift_status == "significant":
        reasons.append("inputs have drifted (PSI at or above " + str(PSI_SIGNIFICANT) + ")")
    if mae_ratio > policy["max_mae_ratio"]:
        reasons.append("forecast error is " + format(mae_ratio, ".2f") + " times the baseline (limit " + str(policy["max_mae_ratio"]) + ")")
    if age_days > policy["max_age_days"]:
        reasons.append("the model is " + str(int(age_days)) + " days old (limit " + str(policy["max_age_days"]) + ")")
    return bool(reasons), reasons