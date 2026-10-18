import pytest
from fastapi.testclient import TestClient

from credit_default_risk.api import Client, app


@pytest.fixture
def client(bundle_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(bundle_path))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def payload() -> dict:
    return dict(Client.model_config["json_schema_extra"]["example"])


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict(client, payload) -> None:
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["default_prediction"] == (body["default_probability"] >= body["threshold"])


def test_predict_rejects_out_of_range_status(client, payload) -> None:
    r = client.post("/predict", json={**payload, "PAY_1": 12})
    assert r.status_code == 422


def test_predict_rejects_sensitive_attributes(client, payload) -> None:
    r = client.post("/predict", json={**payload, "SEX": 1})
    assert r.status_code == 422


def test_predict_rejects_missing_field(client, payload) -> None:
    del payload["LIMIT_BAL"]
    assert client.post("/predict", json=payload).status_code == 422
