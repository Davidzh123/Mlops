from __future__ import annotations

import mlflow
import mlflow.data
import pandas as pd

from mlproject.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MLFLOW_TRACKING_URI,
    TARGET,
)


def setup_experiment() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)
    client = mlflow.MlflowClient()
    if MLFLOW_EXPERIMENT_DESCRIPTION:
        client.set_experiment_tag(
            experiment.experiment_id, "mlflow.note.content", MLFLOW_EXPERIMENT_DESCRIPTION
        )
    for key, value in MLFLOW_EXPERIMENT_TAGS.items():
        client.set_experiment_tag(experiment.experiment_id, key, str(value))


def log_dataset(df: pd.DataFrame, context: str, name: str = "dataset") -> None:
    dataset = mlflow.data.from_pandas(df, source=str(DATA_PATH), targets=TARGET, name=name)  # type: ignore[attr-defined]
    mlflow.log_input(dataset, context=context)
