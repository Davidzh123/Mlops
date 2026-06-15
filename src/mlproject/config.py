from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_PATH = ROOT / "data" / "raw" / "bank_fraud.csv"
MODEL_DIR = ROOT / "models"

TARGET = "is_fraud"

NUMERIC_FEATURES: list[str] = [
    "hour_of_day",
    "is_weekend",
    "is_night_transaction",
    "customer_age",
    "credit_score",
    "account_age_years",
    "account_balance",
    "transaction_amount",
    "num_prev_transactions",
    "transaction_freq_monthly",
    "distance_from_home_km",
    "time_since_last_txn_hrs",
    "is_international",
    "failed_attempts",
    "pin_changed_recently",
]

CATEGORICAL_FEATURES: list[str] = [
    "country",
    "merchant_category",
    "payment_method",
    "device_type",
]

RANDOM_STATE = 42

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "fraude-bancaire-baseline")
MODEL_NAME = os.getenv("MODEL_NAME", "fraude-bancaire-classifier")
