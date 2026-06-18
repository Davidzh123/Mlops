from __future__ import annotations

import subprocess
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

QUALITY_THRESHOLD = 0.65  # seuil sur le roc_auc (metrique de reference, dataset desequilibre)


def task_prepare_data(**context):
    from mlproject.config import DATA_PATH, ROOT

    if DATA_PATH.exists():
        print(f"Donnees deja presentes : {DATA_PATH}")
        return
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_data.py")], check=True)


def task_train(**context):
    from mlproject.train import train

    metrics = train()
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    context["ti"].xcom_push(key="roc_auc", value=metrics["roc_auc"])


def task_check_quality(**context):
    roc_auc = context["ti"].xcom_pull(task_ids="train", key="roc_auc")
    if roc_auc is None or roc_auc < QUALITY_THRESHOLD:
        raise ValueError(f"roc_auc={roc_auc} < seuil {QUALITY_THRESHOLD} : modele rejete")
    print(f"Qualite OK : roc_auc={roc_auc:.3f}")


with DAG(
    dag_id="model_retraining",
    start_date=datetime(2025, 1, 1),
    schedule="0 3 * * 1",  # tous les lundis a 3h
    catchup=False,
    tags=["fraude", "mlops"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    prepare >> train_task >> check
