from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app as app_module

VALID_PAYLOAD = {
    "age": 29.0,
    "salary": 72000.0,
    "score_a": 0.91,
    "score_b": 1.12,
    "tenure": 2.0,
    "department": "Finance",
}


class _ClassifierWithProba:
    def predict_proba(self, X):  # pragma: no cover - presence is what's checked
        raise NotImplementedError


class _ClassifierNoProba:
    pass


def _fake_pipeline(prediction=1, proba=0.8, has_proba=True):
    pipeline = MagicMock()
    pipeline.predict.return_value = [prediction]
    pipeline.predict_proba.return_value = [[1 - proba, proba]]
    classifier = _ClassifierWithProba() if has_proba else _ClassifierNoProba()
    pipeline.named_steps = {"classifier": classifier}
    return pipeline


class TestReadRoot:
    def test_model_loaded_reports_true(self):
        with patch.object(app_module.joblib, "load", return_value=_fake_pipeline()):
            with TestClient(app_module.app) as client:
                resp = client.get("/")

        assert resp.status_code == 200
        assert resp.json() == {
            "status": "online",
            "service": "ML Inference Pipeline",
            "model_loaded": True,
        }

    def test_model_load_failure_reports_false(self):
        with patch.object(app_module.joblib, "load", side_effect=FileNotFoundError("no model")):
            with TestClient(app_module.app) as client:
                resp = client.get("/")

        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is False


class TestPredict:
    def test_success_with_probability(self):
        pipeline = _fake_pipeline(prediction=1, proba=0.8, has_proba=True)
        with patch.object(app_module.joblib, "load", return_value=pipeline):
            with TestClient(app_module.app) as client:
                resp = client.post("/predict", json=VALID_PAYLOAD)

        assert resp.status_code == 200
        body = resp.json()
        assert body["prediction"] == 1
        assert body["probability"] == pytest.approx(0.8)
        pipeline.predict.assert_called_once()

    def test_classifier_without_predict_proba_returns_none(self):
        pipeline = _fake_pipeline(prediction=0, has_proba=False)
        with patch.object(app_module.joblib, "load", return_value=pipeline):
            with TestClient(app_module.app) as client:
                resp = client.post("/predict", json=VALID_PAYLOAD)

        assert resp.status_code == 200
        body = resp.json()
        assert body["prediction"] == 0
        assert body["probability"] is None
        pipeline.predict_proba.assert_not_called()

    def test_pipeline_error_returns_400(self):
        pipeline = _fake_pipeline()
        pipeline.predict.side_effect = ValueError("bad input")
        with patch.object(app_module.joblib, "load", return_value=pipeline):
            with TestClient(app_module.app) as client:
                resp = client.post("/predict", json=VALID_PAYLOAD)

        assert resp.status_code == 400
        assert "Inference error" in resp.json()["detail"]

    def test_model_not_loaded_returns_500(self):
        with patch.object(app_module.joblib, "load", side_effect=RuntimeError("boom")):
            with TestClient(app_module.app) as client:
                resp = client.post("/predict", json=VALID_PAYLOAD)

        assert resp.status_code == 500

    def test_missing_field_returns_422(self):
        payload = dict(VALID_PAYLOAD)
        del payload["age"]
        with patch.object(app_module.joblib, "load", return_value=_fake_pipeline()):
            with TestClient(app_module.app) as client:
                resp = client.post("/predict", json=payload)

        assert resp.status_code == 422
