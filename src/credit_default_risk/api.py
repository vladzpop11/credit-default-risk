"""FastAPI service: GET /health and POST /predict.

The model bundle is loaded once at startup from MODEL_PATH (default models/model.joblib).
The request schema contains only the features the model uses: sex, marital status and
age are deliberately not collected.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from credit_default_risk.config import load_config, project_path
from credit_default_risk.predict import load_bundle, predict

Status = Field(ge=-2, le=9, description="Repayment status: -2..0 paid/revolving, 1-9 months late")
Bill = Field(description="Bill statement amount (NT$); negative means overpaid")
Paid = Field(ge=0, description="Amount paid (NT$)")


class Client(BaseModel):
    """One client's 6-month credit history. Month 1 is the most recent."""

    model_config = ConfigDict(
        extra="forbid",
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

    LIMIT_BAL: float = Field(gt=0, description="Credit limit (NT$)")
    EDUCATION: int = Field(
        ge=0, le=6, description="1 graduate, 2 university, 3 high school, 4+ other"
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


def model_path() -> Path:
    default = project_path(load_config()["paths"]["models_dir"]) / "model.joblib"
    return Path(os.environ.get("MODEL_PATH", default))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle(model_path())
    yield


app = FastAPI(title="Credit default risk", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    bundle = app.state.bundle
    return {"status": "ok", "model_trained_at": bundle["metadata"]["trained_at"]}


@app.post("/predict")
def predict_one(client: Client) -> Prediction:
    bundle = app.state.bundle
    try:
        out = predict(bundle, pd.DataFrame([client.model_dump()])).iloc[0]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Prediction(
        default_probability=float(out["default_probability"]),
        default_prediction=bool(out["default_prediction"]),
        threshold=bundle["threshold"],
        model_trained_at=bundle["metadata"]["trained_at"],
    )
