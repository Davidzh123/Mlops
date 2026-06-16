from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from typing import cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from mlproject.config import MLFLOW_EXPERIMENT, MODEL_DIR, MODEL_NAME, RANDOM_STATE
from mlproject.data import load_data, split
from mlproject.evaluation import log_shap_summary
from mlproject.features import build_preprocessor
from mlproject.tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ModelSpec:
    name: str
    estimator: ClassifierMixin
    param_grid: dict


@dataclass
class FitResult:
    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_score: float
    f1: float
    roc_auc: float
    preds: np.ndarray


def build_model_specs(scale_pos_weight: float) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1
            ),
            param_grid={
                "clf__n_estimators": [100],
                "clf__max_depth": [10, None],
                "clf__min_samples_leaf": [2],
            },
        ),
        ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                n_jobs=1,
            ),
            param_grid={
                "clf__n_estimators": [100],
                "clf__max_depth": [3],
                "clf__learning_rate": [0.1, 0.05],
            },
        ),
        ModelSpec(
            name="lightgbm",
            estimator=LGBMClassifier(
                random_state=RANDOM_STATE, class_weight="balanced", verbose=-1, n_jobs=1
            ),
            param_grid={
                "clf__n_estimators": [100],
                "clf__num_leaves": [31],
                "clf__learning_rate": [0.1, 0.05],
            },
        ),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("clf", estimator)])


def optimize_model(spec, x_train, y_train, x_test, y_test, cv, scoring) -> FitResult:
    logger.info("Optimisation de %s (GridSearchCV, cv=%d, scoring=%s)", spec.name, cv, scoring)
    search = GridSearchCV(
        build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_
    proba = best.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return FitResult(
        name=spec.name,
        best_estimator=best,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),
        f1=float(f1_score(y_test, preds)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        preds=preds,
    )


def log_run_to_mlflow(result, x_test, y_test, cv, scoring, register_as=None) -> None:
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.set_tag("model_family", result.name)
        mlflow.log_param("cv", cv)
        mlflow.log_param("scoring", scoring)
        mlflow.log_params(result.best_params)
        mlflow.log_metrics(
            {f"cv_{scoring}": result.cv_score, "f1": result.f1, "roc_auc": result.roc_auc}
        )

        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        report_text = cast(str, classification_report(y_test, result.preds))
        mlflow.log_dict(report_dict, "classification_report.json")
        mlflow.log_text(report_text, "classification_report.txt")

        log_shap_summary(result.best_estimator, x_test, result.name)

        signature = infer_signature(x_test, result.best_estimator.predict(x_test))
        mlflow.sklearn.log_model(
            result.best_estimator,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )


def train_all(cv: int = 5, scoring: str = "roc_auc", sample: int = 40_000) -> list[FitResult]:
    df = load_data()
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=RANDOM_STATE)

    x_train, x_test, y_train, y_test = split(df)
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    setup_experiment()

    results = [
        optimize_model(spec, x_train, y_train, x_test, y_test, cv, scoring)
        for spec in build_model_specs(scale_pos_weight)
    ]
    results.sort(key=lambda r: r.roc_auc, reverse=True)
    best = results[0]

    print("\n=== Resultats des modeles ===")
    for r in results:
        print(f"{r.name:15s} | cv_{scoring}={r.cv_score:.3f} | f1={r.f1:.3f} | roc_auc={r.roc_auc:.3f}")
    print(f"Meilleur modele : {best.name} (roc_auc={best.roc_auc:.3f})")

    with mlflow.start_run(run_name="compare-models"):
        mlflow.log_param("cv", cv)
        mlflow.log_param("scoring", scoring)
        mlflow.log_param("sample", len(df))
        mlflow.set_tag("best_model", best.name)
        log_dataset(df, context="training", name="bank_fraud")
        for r in results:
            mlflow.log_metric(f"{r.name}_roc_auc", r.roc_auc)
            mlflow.log_metric(f"{r.name}_f1", r.f1)
        for r in results:
            register_as = MODEL_NAME if r.name == best.name else None
            log_run_to_mlflow(r, x_test, y_test, cv, scoring, register_as=register_as)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_estimator, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s", MODEL_DIR / "model.joblib")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", type=str, default="roc_auc")
    parser.add_argument("--sample", type=int, default=40_000)
    args = parser.parse_args()
    train_all(cv=args.cv, scoring=args.scoring, sample=args.sample)


if __name__ == "__main__":
    main()
