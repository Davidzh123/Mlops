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

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "fraude-bancaire")
MODEL_NAME = os.getenv("MODEL_NAME", "fraude-bancaire-classifier")
API_URL = os.getenv("API_URL", "http://api:8000")

MLFLOW_EXPERIMENT_DESCRIPTION = (
    "Detection de fraude bancaire : comparaison de 3 modeles (RandomForest, "
    "XGBoost, LightGBM) optimises par GridSearchCV."
)
MLFLOW_EXPERIMENT_TAGS = {
    "projet": "fraude-bancaire",
    "type": "classification-binaire",
}
