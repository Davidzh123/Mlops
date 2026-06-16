"""Genere les graphiques du README : matrices de confusion + comparaison des metriques.

Entraine les 3 modeles (sans GridSearch, pour aller vite) sur un echantillon,
puis sauvegarde les figures dans docs/.

Usage :
    uv run python scripts/make_report.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from mlproject.config import RANDOM_STATE, ROOT
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor

SAMPLE = 40_000
DOCS = ROOT / "docs"


def pipe(estimator):
    return Pipeline([("preprocessor", build_preprocessor()), ("clf", estimator)])


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    df = load_data().sample(n=SAMPLE, random_state=RANDOM_STATE)
    x_train, x_test, y_train, y_test = split(df)
    spw = (len(y_train) - y_train.sum()) / max(int(y_train.sum()), 1)

    models = {
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", n_estimators=100, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", scale_pos_weight=spw, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", verbose=-1, n_jobs=-1
        ),
    }

    results = {}
    fig_cm, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, est) in zip(axes, models.items()):
        model = pipe(est).fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        results[name] = {
            "roc_auc": roc_auc_score(y_test, proba),
            "f1": f1_score(y_test, preds),
        }
        ConfusionMatrixDisplay(confusion_matrix(y_test, preds)).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name}\nroc_auc={results[name]['roc_auc']:.3f}")
    fig_cm.suptitle("Matrices de confusion (jeu de test)")
    fig_cm.tight_layout()
    fig_cm.savefig(DOCS / "confusion_matrices.png", dpi=110)
    plt.close(fig_cm)

    names = list(results)
    roc = [results[n]["roc_auc"] for n in names]
    f1s = [results[n]["f1"] for n in names]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - 0.2 for i in x], roc, width=0.4, label="ROC AUC")
    ax.bar([i + 0.2 for i in x], f1s, width=0.4, label="F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_title("Comparaison des modeles")
    ax.legend()
    for i, (r, f) in enumerate(zip(roc, f1s)):
        ax.text(i - 0.2, r + 0.01, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "metrics_comparison.png", dpi=110)
    plt.close(fig)

    print("Figures generees dans docs/ :")
    print("  - confusion_matrices.png")
    print("  - metrics_comparison.png")
    for n in names:
        print(f"  {n:15s} roc_auc={results[n]['roc_auc']:.3f}  f1={results[n]['f1']:.3f}")


if __name__ == "__main__":
    main()
