"""Automated tests for the SuperKart API and the shared feature module.

Run from the repository root:  python -m unittest discover -s tests -v
"""
import importlib
import io
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import superkart_features as sf  # noqa: E402

SAMPLE = {"Product_Id": "FD6114", "Product_Weight": 12.66, "Product_Sugar_Content": "Low Sugar", "Product_Allocated_Area": 0.027,
          "Product_Type": "Frozen Foods", "Product_MRP": 117.08, "Store_Establishment_Year": 2009, "Store_Size": "Medium",
          "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type2"}


def load_backend(**env):
    """Import a fresh copy of the backend with the given environment variables."""
    for key in ("API_KEY", "RATE_LIMIT_PER_MINUTE"):
        os.environ.pop(key, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    sys.modules.pop("app", None)
    return importlib.import_module("app")


class FeatureTests(unittest.TestCase):
    def test_engineer_features_fixes_labels_and_builds_features(self):
        raw = pd.DataFrame([dict(SAMPLE, Product_Sugar_Content="reg", Store_Id="OUT001")])
        out = sf.engineer_features(raw)
        self.assertEqual(out.loc[0, "Product_Sugar_Content"], "Regular")
        self.assertEqual(out.loc[0, "Product_Id_Prefix"], "FD")
        self.assertEqual(out.loc[0, "Store_Age"], sf.REFERENCE_YEAR - 2009)
        for column in ["Product_Id", "Store_Id", "Store_Establishment_Year"]:
            self.assertNotIn(column, out.columns)

    def test_validation_reports_every_problem(self):
        bad = pd.DataFrame([dict(SAMPLE, Product_MRP="abc", Store_Type="Mall", Product_Id="1234")])
        problems = sf.validate_records(bad)
        self.assertTrue(any("Product_MRP" in p for p in problems))
        schema = {"categories": {"Store_Type": ["Food Mart"]}, "numeric": {}, "product_id_prefixes": ["FD"]}
        problems = sf.validate_records(bad, schema)
        self.assertTrue(any("Store_Type" in p for p in problems))
        self.assertTrue(any("Product_Id" in p for p in problems))

    def test_conformal_half_widths_cover_the_requested_share(self):
        rng = np.random.default_rng(0)
        residuals = np.abs(rng.normal(0, 100, 2000))
        widths = sf.conformal_half_widths(np.array(["A"] * 2000), residuals, 0.9)
        self.assertGreaterEqual((residuals <= widths["half_width"]["A"]).mean(), 0.9)

    def test_psi_is_zero_for_identical_data_and_large_for_shifted_data(self):
        rng = np.random.default_rng(1)
        raw = pd.DataFrame(dict(SAMPLE, Product_Weight=rng.normal(12, 2, 500), Product_Allocated_Area=rng.uniform(0.01, 0.2, 500),
                                Product_MRP=rng.normal(150, 30, 500)), index=range(500))
        reference = sf.build_drift_reference(raw)
        self.assertEqual(sf.drift_report(reference, raw)["overall_status"], "stable")
        shifted = raw.assign(Product_MRP=raw["Product_MRP"] + 80)
        self.assertEqual(sf.drift_report(reference, shifted)["overall_status"], "significant")

    def test_retraining_policy(self):
        policy = {"max_mae_ratio": 1.25, "max_age_days": 90}
        self.assertFalse(sf.needs_retraining("stable", 1.0, 10, policy)[0])
        self.assertTrue(sf.needs_retraining("significant", 1.0, 10, policy)[0])
        self.assertTrue(sf.needs_retraining("stable", 1.5, 10, policy)[0])
        self.assertTrue(sf.needs_retraining("stable", 1.0, 200, policy)[0])


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = load_backend()
        cls.client = cls.backend.app.test_client()

    def test_health(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["model_loaded"])

    def test_single_forecast_has_prediction_and_range(self):
        response = self.client.post("/v1/forecast", json=SAMPLE)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertGreater(body["Predicted_Product_Store_Sales_Total"], 0)
        if self.backend.INTERVAL:
            self.assertLessEqual(body["Lower_Bound"], body["Predicted_Product_Store_Sales_Total"])
            self.assertGreaterEqual(body["Upper_Bound"], body["Predicted_Product_Store_Sales_Total"])

    def test_api_matches_the_model_directly(self):
        direct = float(self.backend.model.predict(sf.prepare_features(pd.DataFrame([SAMPLE]), self.backend.SCHEMA))[0])
        api = self.client.post("/v1/forecast", json=SAMPLE).get_json()["Predicted_Product_Store_Sales_Total"]
        self.assertAlmostEqual(api, round(direct, 2), places=2)

    def test_bad_inputs_are_rejected_with_400(self):
        cases = [{"Product_Id": "FD1"}, dict(SAMPLE, Product_MRP="abc"), dict(SAMPLE, Store_Type="Mall"),
                 dict(SAMPLE, Product_MRP=-5), dict(SAMPLE, Product_Id="12")]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/v1/forecast", json=payload).status_code, 400)
        self.assertEqual(self.client.post("/v1/forecast", data="{not json", content_type="application/json").status_code, 400)

    def test_batch(self):
        csv = pd.DataFrame([SAMPLE, dict(SAMPLE, Product_Id="NC1180")]).to_csv(index=False).encode()
        response = self.client.post("/v1/forecastbatch", data={"file": (io.BytesIO(csv), "b.csv")}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["count"], 2)
        empty = self.client.post("/v1/forecastbatch", data={"file": (io.BytesIO(b"Product_Id\n"), "e.csv")}, content_type="multipart/form-data")
        self.assertEqual(empty.status_code, 400)

    def test_unknown_route_returns_json_404(self):
        response = self.client.get("/nope")
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.get_json())

    def test_schema_and_metadata_routes(self):
        self.assertEqual(self.client.get("/v1/schema").status_code, 200)
        self.assertEqual(self.client.get("/v1/metadata").status_code, 200)


class SecurityTests(unittest.TestCase):
    def test_api_key_is_enforced_but_health_stays_open(self):
        backend = load_backend(API_KEY="secret")
        client = backend.app.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.post("/v1/forecast", json=SAMPLE).status_code, 401)
        self.assertEqual(client.post("/v1/forecast", json=SAMPLE, headers={"X-API-Key": "wrong"}).status_code, 401)
        self.assertEqual(client.post("/v1/forecast", json=SAMPLE, headers={"X-API-Key": "secret"}).status_code, 200)

    def test_rate_limit_returns_429(self):
        backend = load_backend(RATE_LIMIT_PER_MINUTE=3)
        client = backend.app.test_client()
        codes = [client.post("/v1/forecast", json=SAMPLE).status_code for _ in range(5)]
        self.assertEqual(codes[:3], [200, 200, 200])
        self.assertEqual(codes[3:], [429, 429])

    def test_api_answers_503_without_a_model_and_recovers(self):
        backend = load_backend()
        client = backend.app.test_client()
        saved = backend.model
        backend.model = None
        try:
            self.assertEqual(client.get("/").status_code, 503)
            self.assertEqual(client.post("/v1/forecast", json=SAMPLE).status_code, 503)
        finally:
            backend.model = saved
        self.assertEqual(client.get("/").status_code, 200)


class MetadataTests(unittest.TestCase):
    def test_metadata_file_is_consistent_with_the_model_file(self):
        with open(os.path.join(BACKEND, "model_metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.assertTrue(os.path.exists(os.path.join(BACKEND, meta["model_file"])))
        self.assertIn("schema", meta)


if __name__ == "__main__":
    unittest.main()
