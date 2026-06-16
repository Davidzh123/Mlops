from __future__ import annotations

import os

import httpx

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

TRANSACTION = {
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


def main() -> None:
    response = httpx.post(f"{API_URL}/predict", json=TRANSACTION, timeout=10.0)
    response.raise_for_status()
    result = response.json()
    label = "FRAUDE" if result["prediction"] == 1 else "LEGITIME"
    print(f"Prediction : {result['prediction']} ({label})")
    print(f"Probabilite de fraude : {result['probability']:.2%}")


if __name__ == "__main__":
    main()
