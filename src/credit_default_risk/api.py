"""FastAPI service: GET/HEAD /health and POST /predict.

The model bundle is loaded once at startup from MODEL_PATH (default models/model.joblib).
It is a pickle, so it must be a trusted file; set MODEL_SHA256 to refuse any other file
before it is unpickled. /health reports whether the loaded file is the one evaluated on
the test set (reports/test_metrics.json).

The request schema contains only the features the model uses (sex, marital status and age
are deliberately not collected) and is strict: exact JSON types, realistic ranges, no
unknown or duplicate keys, a JSON content type and a small body. Production deployments
should still sit behind a reverse proxy for TLS, timeouts on slow clients and rate limits.
"""

import json
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from credit_default_risk import __version__, artifacts
from credit_default_risk.config import load_config
from credit_default_risk.predict import load_bundle, predict

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 16_384
MAX_ERRORS = 20
MAX_AMOUNT = 100_000_000  # NT$; the training data's largest amount is about 1.7 million

Status = Field(ge=-2, le=9, description="Repayment status: -2..0 paid/revolving, 1-9 months late")
Bill = Field(
    ge=-MAX_AMOUNT, le=MAX_AMOUNT, description="Bill statement amount (NT$); negative = overpaid"
)
Paid = Field(ge=0, le=MAX_AMOUNT, description="Amount paid (NT$)")


class Client(BaseModel):
    """One client's 6-month credit history. Month 1 (September 2005) is the most recent.

    PAY_1 is the column the UCI file calls PAY_0.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,  # no "50000" strings or true/false for numbers
        # JSON parsers accept NaN/Infinity literals; reject them.
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "LIMIT_BAL": 50000, "EDUCATION": 2,
                "PAY_1": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 48000, "BILL_AMT2": 47000, "BILL_AMT3": 46000,
                "BILL_AMT4": 30000, "BILL_AMT5": 28000, "BILL_AMT6": 25000,
                "PAY_AMT1": 0, "PAY_AMT2": 1500, "PAY_AMT3": 1500,
                "PAY_AMT4": 1200, "PAY_AMT5": 1000, "PAY_AMT6": 1000,
            }
        },
    )  # fmt: skip

    LIMIT_BAL: float = Field(ge=1_000, le=10_000_000, description="Credit limit (NT$)")
    EDUCATION: int = Field(
        ge=0, le=6, description="1 graduate, 2 university, 3 high school, 0/4/5/6 other"
    )
    PAY_1: int = Status
    PAY_2: int = Status
    PAY_3: int = Status
    PAY_4: int = Status
    PAY_5: int = Status
    PAY_6: int = Status
    BILL_AMT1: float = Bill
    BILL_AMT2: float = Bill
    BILL_AMT3: float = Bill
    BILL_AMT4: float = Bill
    BILL_AMT5: float = Bill
    BILL_AMT6: float = Bill
    PAY_AMT1: float = Paid
    PAY_AMT2: float = Paid
    PAY_AMT3: float = Paid
    PAY_AMT4: float = Paid
    PAY_AMT5: float = Paid
    PAY_AMT6: float = Paid


class Prediction(BaseModel):
    default_probability: float
    default_prediction: bool
    threshold: float
    model_trained_at: str


def error(status: int, type_: str, msg: str) -> JSONResponse:
    """Errors always use FastAPI's list shape: {"detail": [{"type", "loc", "msg"}]}."""
    return JSONResponse(
        status_code=status, content={"detail": [{"type": type_, "loc": ["body"], "msg": msg}]}
    )


class DuplicateKeyError(ValueError):
    pass


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise DuplicateKeyError(f"duplicate keys: {duplicates}")
    return dict(pairs)


class RequestGuard:
    """ASGI middleware for POST bodies: JSON content type, size limit, no duplicate keys."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        content_type = headers.get(b"content-type", b"").split(b";")[0].strip().lower()
        if content_type != b"application/json":
            msg = "Content-Type must be application/json"
            return await error(415, "unsupported_media_type", msg)(scope, receive, send)
        declared = headers.get(b"content-length", b"0")
        if not declared.isdigit() or int(declared) > MAX_BODY_BYTES:
            return await self._too_large(scope, receive, send)

        body, more = b"", True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            more = message.get("more_body", False)
            if len(body) > MAX_BODY_BYTES:
                return await self._too_large(scope, receive, send)
        try:
            json.loads(body, object_pairs_hook=_reject_duplicates)
        except DuplicateKeyError as exc:
            return await error(422, "duplicate_key", str(exc))(scope, receive, send)
        except ValueError:
            pass  # invalid JSON: FastAPI reports it as a validation error

        replayed = False

        async def replay() -> dict:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    @staticmethod
    async def _too_large(scope: dict, receive: Callable, send: Callable) -> None:
        msg = f"request body larger than {MAX_BODY_BYTES} bytes"
        await error(413, "body_too_large", msg)(scope, receive, send)


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", artifacts.model_path(load_config())))


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = model_path()
    bundle = load_bundle(path, expected_sha256=os.environ.get("MODEL_SHA256"))
    # One thread per prediction: the server already handles requests concurrently.
    bundle["pipeline"].set_params(model__n_jobs=1)
    app.state.bundle = bundle
    app.state.model_sha256 = artifacts.sha256(path)
    app.state.is_evaluated = app.state.model_sha256 == artifacts.evaluated_model_sha(load_config())
    yield


app = FastAPI(
    title="Credit default risk", version=__version__, lifespan=lifespan, redirect_slashes=False
)
app.add_middleware(RequestGuard)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # The default handler echoes the rejected input, which fails to serialise for
    # NaN/Infinity (a 500 instead of a 422), repeats client data back and can be huge.
    detail = [{k: e[k] for k in ("type", "loc", "msg") if k in e} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail[:MAX_ERRORS]})


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict:
    return {
        "status": "ok",
        "model_trained_at": app.state.bundle["metadata"]["trained_at"],
        "model_sha256": app.state.model_sha256,
        "evaluated_model": app.state.is_evaluated,
    }


@app.post("/predict")
def predict_one(client: Client) -> Prediction:
    bundle = app.state.bundle
    try:
        out = predict(bundle, pd.DataFrame([client.model_dump()])).iloc[0]
    except Exception:
        logger.exception("prediction failed")
        return error(500, "prediction_failed", "the input could not be scored")
    return Prediction(
        default_probability=float(out["default_probability"]),
        default_prediction=bool(out["default_prediction"]),
        threshold=bundle["threshold"],
        model_trained_at=bundle["metadata"]["trained_at"],
    )
