"""DAG Airflow - pipeline de re-entrainement du modele.

Seance 17 - TP Airflow
Pipeline : preparation des donnees -> entrainement -> controle qualite.
Le garde-fou utilise le roc_auc (metrique de reference, dataset desequilibre).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# roc_auc minimal pour que le pipeline soit considere comme reussi.
QUALITY_THRESHOLD = 0.65

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    import subprocess
    import sys

    from mlproject.config import DATA_PATH, ROOT

    if DATA_PATH.exists():
        logger.info("Donnees deja presentes : %s", DATA_PATH)
        return
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_data.py")], check=True)


def task_train(**context) -> None:
    from mlproject.train import train

    metrics = train()
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    context["ti"].xcom_push(key="roc_auc", value=metrics["roc_auc"])


def task_check_quality(**context) -> None:
    roc_auc = context["ti"].xcom_pull(task_ids="train", key="roc_auc")
    if roc_auc is None or roc_auc < QUALITY_THRESHOLD:
        raise ValueError(f"roc_auc={roc_auc} < seuil {QUALITY_THRESHOLD} : modele rejete")
    logger.info("Qualite OK : roc_auc=%.3f", roc_auc)


with DAG(
    dag_id="model_retraining",
    description="Prepare les donnees, reentraine le modele et controle sa qualite",
    schedule="0 3 * * 1",  # tous les lundis a 3h
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["fraude", "training"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    prepare >> train_task >> check
