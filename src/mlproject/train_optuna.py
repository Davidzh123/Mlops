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
import optuna
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
from sklearn.model_selection import cross_val_score
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
optuna.logging.set_verbosity(optuna.logging.WARNING)

FAMILIES = ["random_forest", "xgboost", "lightgbm"]


@dataclass
class FamilyResult:
    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_score: float
    f1: float
    roc_auc: float
    preds: np.ndarray


def suggest_params(family: str, trial: optuna.Trial) -> dict:
    if family == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
    }


def build_estimator(family: str, params: dict, scale_pos_weight: float) -> ClassifierMixin:
    if family == "random_forest":
        return RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1, **params
        )
    if family == "xgboost":
        return XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=1,
            **params,
        )
    return LGBMClassifier(
        random_state=RANDOM_STATE, class_weight="balanced", verbose=-1, n_jobs=1, **params
    )


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("clf", estimator)])


def objective(trial, family, x_train, y_train, cv, scale_pos_weight) -> float:
    params = suggest_params(family, trial)
    pipeline = build_pipeline(build_estimator(family, params, scale_pos_weight))
    scores = cross_val_score(pipeline, x_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
    return float(scores.mean())


def optimize_family(family, x_train, y_train, x_test, y_test, n_trials, cv, spw) -> FamilyResult:
    logger.info("Optuna sur %s (n_trials=%d, cv=%d)", family, n_trials, cv)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(
        lambda t: objective(t, family, x_train, y_train, cv, spw), n_trials=n_trials
    )

    best = build_pipeline(build_estimator(family, study.best_params, spw))
    best.fit(x_train, y_train)
    proba = best.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return FamilyResult(
        name=family,
        best_estimator=best,
        best_params=study.best_params,
        cv_score=float(study.best_value),
        f1=float(f1_score(y_test, preds)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        preds=preds,
    )


def log_to_mlflow(result, x_test, y_test, n_trials, cv, register_as=None) -> None:
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.set_tag("model_family", result.name)
        mlflow.set_tag("search_method", "optuna-tpe")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv", cv)
        mlflow.log_params(result.best_params)
        mlflow.log_metrics(
            {"cv_roc_auc": result.cv_score, "f1": result.f1, "roc_auc": result.roc_auc}
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


def optimize(n_trials: int = 20, cv: int = 3, sample: int = 40_000) -> list[FamilyResult]:
    df = load_data()
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=RANDOM_STATE)

    x_train, x_test, y_train, y_test = split(df)
    n_pos = int(y_train.sum())
    spw = (len(y_train) - n_pos) / max(n_pos, 1)

    setup_experiment()

    results = [
        optimize_family(f, x_train, y_train, x_test, y_test, n_trials, cv, spw) for f in FAMILIES
    ]
    results.sort(key=lambda r: r.roc_auc, reverse=True)
    best = results[0]

    print("\n=== Resultats Optuna ===")
    for r in results:
        print(f"{r.name:15s} | cv_roc_auc={r.cv_score:.3f} | f1={r.f1:.3f} | roc_auc={r.roc_auc:.3f}")
    print(f"Meilleur modele : {best.name} (roc_auc={best.roc_auc:.3f})")

    with mlflow.start_run(run_name="optuna-compare"):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv", cv)
        mlflow.log_param("sample", len(df))
        mlflow.set_tag("best_model", best.name)
        log_dataset(df, context="training", name="bank_fraud")
        for r in results:
            mlflow.log_metric(f"{r.name}_roc_auc", r.roc_auc)
            mlflow.log_metric(f"{r.name}_f1", r.f1)
        for r in results:
            register_as = MODEL_NAME if r.name == best.name else None
            log_to_mlflow(r, x_test, y_test, n_trials, cv, register_as=register_as)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_estimator, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s", MODEL_DIR / "model.joblib")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--sample", type=int, default=40_000)
    args = parser.parse_args()
    optimize(n_trials=args.n_trials, cv=args.cv, sample=args.sample)


if __name__ == "__main__":
    main()
