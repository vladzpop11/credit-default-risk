import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from credit_default_risk.api import MAX_BODY_BYTES, MAX_ERRORS, Client, app

EXAMPLE_FILE = Path(__file__).parent / "example_client.json"


@pytest.fixture
def client(bundle_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(bundle_path))
    monkeypatch.delenv("MODEL_SHA256", raising=False)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def payload() -> dict:
    return dict(Client.model_config["json_schema_extra"]["example"])


def post_raw(client, body: str, content_type: str = "application/json"):
    return client.post("/predict", content=body, headers={"Content-Type": content_type})


def assert_error_shape(r, status: int) -> None:
    assert r.status_code == status
    detail = r.json()["detail"]
    assert isinstance(detail, list) and detail
    assert all(set(e) <= {"type", "loc", "msg"} for e in detail)


def test_example_file_matches_the_schema_example(payload) -> None:
    assert json.loads(EXAMPLE_FILE.read_text()) == payload


def test_health(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert len(body["model_sha256"]) == 64
    assert body["evaluated_model"] is False  # the test bundle is not the evaluated model


def test_health_supports_head(client) -> None:
    assert client.head("/health").status_code == 200


def test_root_redirects_to_docs(client) -> None:
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/docs"


def test_trailing_slash_is_not_redirected(client, payload) -> None:
    r = client.post("/predict/", json=payload, follow_redirects=False)
    assert r.status_code == 404


def test_predict(client, payload) -> None:
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "default_probability",
        "default_prediction",
        "threshold",
        "model_trained_at",
    }
    assert 0 <= body["default_probability"] <= 1
    assert body["default_prediction"] == (body["default_probability"] >= body["threshold"])


def test_ints_are_accepted_for_float_fields(client, payload) -> None:
    assert isinstance(payload["LIMIT_BAL"], int)
    assert client.post("/predict", json=payload).status_code == 200
    assert client.post("/predict", json={**payload, "BILL_AMT1": 48000.5}).status_code == 200


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("PAY_1", 12),  # out of range
        ("PAY_1", True),  # booleans are not numbers
        ("PAY_1", "2"),  # no string coercion
        ("PAY_1", 2.0),  # an int field takes ints only
        ("LIMIT_BAL", "50000"),
        ("LIMIT_BAL", 1e-305),  # tiny limits would overflow utilisation
        ("LIMIT_BAL", 500),
        ("LIMIT_BAL", 1e12),
        ("BILL_AMT1", 1e300),
        ("PAY_AMT1", -1),
        ("EDUCATION", False),
    ],
)
def test_predict_rejects_bad_values(client, payload, field, value) -> None:
    assert_error_shape(client.post("/predict", json={**payload, field: value}), 422)


def test_predict_rejects_sensitive_attributes(client, payload) -> None:
    assert_error_shape(client.post("/predict", json={**payload, "SEX": 1}), 422)


def test_predict_rejects_the_raw_uci_column_name(client, payload) -> None:
    body = {**payload, "PAY_0": payload.pop("PAY_1")}
    assert_error_shape(client.post("/predict", json=body), 422)


def test_predict_rejects_missing_field(client, payload) -> None:
    del payload["LIMIT_BAL"]
    assert_error_shape(client.post("/predict", json=payload), 422)


@pytest.mark.parametrize("field", ["LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"])
@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_predict_rejects_non_finite_numbers(client, payload, field, literal) -> None:
    # NaN/Infinity are not valid JSON but are accepted by Python's JSON parser.
    body = json.dumps(payload).replace(f'"{field}": {payload[field]}', f'"{field}": {literal}')
    assert literal in body
    assert_error_shape(post_raw(client, body), 422)


def test_errors_do_not_echo_the_input(client, payload) -> None:
    secret = "do-not-echo-123"
    r = client.post("/predict", json={**payload, "LIMIT_BAL": secret})
    assert_error_shape(r, 422)
    assert secret not in r.text


def test_errors_are_capped(client, payload) -> None:
    extra = {f"k{i}": 0 for i in range(200)}
    r = client.post("/predict", json={**payload, **extra})
    assert_error_shape(r, 422)
    assert len(r.json()["detail"]) <= MAX_ERRORS


def test_predict_rejects_duplicate_keys(client, payload) -> None:
    body = json.dumps(payload)[:-1] + ', "PAY_1": 9}'
    r = post_raw(client, body)
    assert_error_shape(r, 422)
    assert "PAY_1" in r.json()["detail"][0]["msg"]


@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", ""])
def test_predict_requires_json_content_type(client, payload, content_type) -> None:
    assert_error_shape(post_raw(client, json.dumps(payload), content_type), 415)


def test_predict_accepts_json_with_charset(client, payload) -> None:
    r = post_raw(client, json.dumps(payload), "application/json; charset=utf-8")
    assert r.status_code == 200


def test_predict_rejects_large_bodies(client, payload) -> None:
    body = json.dumps({**payload, "padding": "x" * MAX_BODY_BYTES})
    assert_error_shape(post_raw(client, body), 413)


def test_invalid_json_is_a_validation_error(client) -> None:
    assert_error_shape(post_raw(client, '{"LIMIT_BAL": 5'), 422)


def test_startup_refuses_a_model_with_the_wrong_hash(bundle_path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PATH", str(bundle_path))
    monkeypatch.setenv("MODEL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="refusing to unpickle"), TestClient(app):
        pass
