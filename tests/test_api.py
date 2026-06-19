from __future__ import annotations

from fastapi.testclient import TestClient

from mlproject.api import app

EXAMPLE = {
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


def test_health() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_predict() -> None:
    with TestClient(app) as client:
        resp = client.post("/predict", json=EXAMPLE)
        assert resp.status_code in (200, 503)
