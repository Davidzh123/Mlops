from __future__ import annotations

import numpy as np
import pandas as pd

from mlproject.config import DATA_PATH

N = 3000


def main() -> None:
    if DATA_PATH.exists():
        print(f"{DATA_PATH} existe deja, generation ignoree (rien ecrase).")
        return

    rng = np.random.default_rng(42)
    n = N
    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:08d}" for i in range(n)],
            "customer_id": [f"CUST{i:06d}" for i in range(n)],
            "transaction_date": "2023-01-01",
            "transaction_time": "12:00:00",
            "hour_of_day": rng.integers(0, 24, n),
            "is_weekend": rng.integers(0, 2, n),
            "is_night_transaction": rng.integers(0, 2, n),
            "country": rng.choice(["USA", "UK", "France", "Canada"], n),
            "city": rng.choice(["A", "B", "C"], n),
            "merchant_category": rng.choice(["Grocery", "Electronics", "Utilities", "Clothing"], n),
            "payment_method": rng.choice(["Credit Card", "Debit Card", "Crypto", "Cheque"], n),
            "device_type": rng.choice(["Mobile", "Desktop", "POS Terminal", "ATM"], n),
            "customer_age": rng.integers(18, 90, n),
            "credit_score": rng.integers(300, 851, n),
            "account_age_years": rng.uniform(0, 30, n).round(1),
            "account_balance": rng.uniform(0, 50000, n).round(2),
            "transaction_amount": rng.uniform(1, 5000, n).round(2),
            "num_prev_transactions": rng.integers(0, 300, n),
            "transaction_freq_monthly": rng.integers(0, 50, n),
            "distance_from_home_km": rng.uniform(0, 500, n).round(1),
            "time_since_last_txn_hrs": rng.uniform(0, 72, n).round(2),
            "is_international": rng.integers(0, 2, n),
            "failed_attempts": rng.integers(0, 5, n),
            "pin_changed_recently": rng.integers(0, 2, n),
            "fraud_type": "None",
        }
    )

    logits = (
        0.0008 * df["transaction_amount"]
        + 0.5 * df["failed_attempts"]
        + 0.8 * df["is_international"]
        - 3.0
    )
    prob = 1 / (1 + np.exp(-logits))
    df["is_fraud"] = (rng.uniform(0, 1, n) < prob).astype(int)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(
        f"Echantillon synthetique ecrit : {DATA_PATH} "
        f"({len(df)} lignes, {df['is_fraud'].mean() * 100:.1f}% fraude)"
    )


if __name__ == "__main__":
    main()
