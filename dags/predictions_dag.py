from __future__ import annotations

import json
import os
from datetime import datetime

import httpx
from airflow import DAG
from airflow.operators.python import PythonOperator

API_URL = os.environ.get("API_URL", "http://api:8000")
N_PREDICTIONS = 20


def task_send_predictions(**context):
    from mlproject.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    from mlproject.data import load_data

    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    sample = load_data()[cols].sample(n=N_PREDICTIONS)

    with httpx.Client(base_url=API_URL, timeout=10.0) as client:
        client.get("/health").raise_for_status()
        for _, row in sample.iterrows():
            payload = json.loads(row.to_json())  # types JSON natifs (pas de numpy)
            response = client.post("/predict", json=payload)
            response.raise_for_status()
    print(f"{N_PREDICTIONS} previsions envoyees a {API_URL}")


with DAG(
    dag_id="daily_predictions",
    start_date=datetime(2025, 1, 1),
    schedule="0 10 * * *",  # tous les jours a 10h
    catchup=False,
    tags=["fraude", "mlops"],
) as dag:
    send = PythonOperator(task_id="send_predictions", python_callable=task_send_predictions)
