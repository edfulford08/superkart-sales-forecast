"""Wide randomized hyperparameter search for the SuperKart models, with an on-disk cache.

Use it from the notebook (import superkart_search) or from the command line to run the slow search outside the
notebook:  python superkart_search.py --data SuperKart.csv --out search_cache.json --iterations 40
A cached result is reused only when the data, the settings and the library versions are unchanged.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# the shared feature module lives in the backend folder of the repository
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
import superkart_features as sf  # noqa: E402

try:
    from xgboost import XGBRegressor
except ImportError:  # the search for XGBoost is skipped when xgboost is missing
    XGBRegressor = None

SEARCH_SPACES = {
    "Random Forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 8, 12, 16, 24],
        "min_samples_leaf": [1, 2, 3, 4, 5, 8],
        "max_features": [0.3, 0.5, 0.7, 1.0],
        "max_samples": [None, 0.7, 0.9],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300, 500, 800],
        "max_depth": [3, 4, 5, 6, 7, 8, 10],
        "learning_rate": stats.loguniform(0.01, 0.3),
        "subsample": stats.uniform(0.5, 0.5),
        "colsample_bytree": stats.uniform(0.5, 0.5),
        "min_child_weight": [1, 2, 3, 5, 8, 10],
        "reg_lambda": stats.loguniform(0.1, 20),
    },
}


def build_preprocessor():
    return make_column_transformer(
        (StandardScaler(), sf.NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), sf.CATEGORICAL_FEATURES),
    )


def build_estimator(name, params=None, seed=42):
    params = dict(params or {})
    if name == "Random Forest":
        return make_pipeline(build_preprocessor(), RandomForestRegressor(random_state=seed, n_jobs=-1, **params))
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed")
    return make_pipeline(build_preprocessor(), XGBRegressor(random_state=seed, n_jobs=-1, objective="reg:squarederror", **params))


def _prefix(name):
    return "randomforestregressor__" if name == "Random Forest" else "xgbregressor__"


def data_fingerprint(X, y, n_iter, seed, n_splits):
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(X, index=False).values.tobytes())
    digest.update(np.asarray(y, dtype=float).tobytes())
    digest.update(json.dumps([n_iter, seed, n_splits, sklearn.__version__, np.__version__, pd.__version__], sort_keys=True).encode())
    return digest.hexdigest()


def _clean(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def run_search(X, y, n_iter=40, seed=42, n_splits=5, cache_path=None, models=("Random Forest", "XGBoost"), n_jobs=-1):
    """Randomized search for every model; returns {name: {best_params, best_cv_rmse, top, seconds, from_cache}}."""
    fingerprint = data_fingerprint(X, y, n_iter, seed, n_splits)
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
    if cache.get("fingerprint") != fingerprint:
        cache = {"fingerprint": fingerprint, "models": {}}
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = {}
    for name in models:
        if name == "XGBoost" and XGBRegressor is None:
            continue
        if name in cache["models"]:
            entry = dict(cache["models"][name])
            entry["from_cache"] = True
            out[name] = entry
            continue
        started = time.time()
        pipe = build_estimator(name, seed=seed)
        space = {_prefix(name) + k: v for k, v in SEARCH_SPACES[name].items()}
        search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="neg_root_mean_squared_error", cv=cv,
                                    random_state=seed, n_jobs=n_jobs, refit=False)
        search.fit(X, y)
        table = pd.DataFrame(search.cv_results_)
        table["rmse"] = -table["mean_test_score"]
        table = table.sort_values("rmse").reset_index(drop=True)
        best_params = {k.split("__", 1)[1]: _clean(v) for k, v in search.best_params_.items()}
        top = [{"rank": int(i + 1), "rmse": float(r["rmse"]),
                "params": {k.split("__", 1)[1]: _clean(v) for k, v in r["params"].items()}} for i, r in table.head(5).iterrows()]
        entry = {"best_params": best_params, "best_cv_rmse": float(table.loc[0, "rmse"]), "top": top,
                 "n_iter": int(n_iter), "seconds": round(time.time() - started, 1)}
        cache["models"][name] = entry
        if cache_path:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
            except OSError as err:
                print("Could not write the search cache:", err)
        out[name] = dict(entry, from_cache=False)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="SuperKart.csv")
    parser.add_argument("--out", default="search_cache.json")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = pd.read_csv(args.data)
    data = sf.engineer_features(raw)
    target = "Product_Store_Sales_Total"
    result = run_search(data[sf.MODEL_FEATURES], data[target], args.iterations, args.seed, cache_path=args.out)
    for name, entry in result.items():
        print(name, "best CV RMSE", round(entry["best_cv_rmse"], 2), entry["best_params"])


if __name__ == "__main__":
    main()