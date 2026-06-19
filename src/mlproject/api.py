from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlproject.config import MODEL_DIR

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = MODEL_DIR / "model.joblib"
    if model_path.exists():
        ml["model"] = joblib.load(model_path)
    yield
    ml.clear()


app = FastAPI(
    title="API de detection de fraude bancaire",
    version="1.0.0",
    lifespan=lifespan,
)


class Features(BaseModel):
    hour_of_day: int = Field(..., ge=0, le=23)
    is_weekend: int = Field(..., ge=0, le=1)
    is_night_transaction: int = Field(..., ge=0, le=1)
    customer_age: int = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    account_age_years: float = Field(..., ge=0)
    account_balance: float
    transaction_amount: float = Field(..., ge=0)
    num_prev_transactions: int = Field(..., ge=0)
    transaction_freq_monthly: int = Field(..., ge=0)
    distance_from_home_km: float = Field(..., ge=0)
    time_since_last_txn_hrs: float = Field(..., ge=0)
    is_international: int = Field(..., ge=0, le=1)
    failed_attempts: int = Field(..., ge=0)
    pin_changed_recently: int = Field(..., ge=0, le=1)
    country: str
    merchant_category: str
    payment_method: str
    device_type: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "hour_of_day": 2,
                    "is_weekend": 0,
                    "is_night_transaction": 1,
                    "customer_age": 29,
                    "credit_score": 520,
                    "account_age_years": 1.2,
                    "account_balance": 80.0,
                    "transaction_amount": 4200.0,
                    "num_prev_transactions": 12,
                    "transaction_freq_monthly": 3,
                    "distance_from_home_km": 320.0,
                    "time_since_last_txn_hrs": 0.5,
                    "is_international": 1,
                    "failed_attempts": 3,
                    "pin_changed_recently": 1,
                    "country": "France",
                    "merchant_category": "Electronics",
                    "payment_method": "Credit Card",
                    "device_type": "Mobile",
                }
            ]
        }
    }


class PredictionOut(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in ml}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


@app.get("/model-info")
def model_info() -> dict:
    return {"version": os.environ.get("MODEL_VERSION", "unknown"), "model_loaded": "model" in ml}
