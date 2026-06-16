"""Valide la chaine de preparation des donnees : config.py, data.py, features.py.

Lance les trois composants et verifie :
  - config.py  : la cible et les colonnes declarees existent, la cible est binaire
  - data.py    : le CSV se charge et le split train/test fonctionne (stratifie)
  - features.py: le ColumnTransformer s'ajuste et transforme les donnees

Usage :
    uv run python scripts/preview_data.py
"""

from __future__ import annotations

import sys

import pandas as pd

from mlproject.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

OK = "[ OK ]"
KO = "[FAIL]"


def main() -> int:
    errors: list[str] = []

    print("=" * 70)
    print("VALIDATION : config.py + data.py + features.py")
    print("=" * 70)

    # --- 1. config.py + data.py : chargement -------------------------------
    df = load_data()
    print(f"\n[1] data.py  -> CSV charge : {len(df):,} lignes, {df.shape[1]} colonnes")

    print("\n[2] config.py -> verification des colonnes declarees")
    if TARGET not in df.columns:
        errors.append(f"cible '{TARGET}' absente du CSV")
        print(f"  {KO} cible '{TARGET}' absente")
    else:
        n_classes = df[TARGET].nunique()
        binaire = sorted(df[TARGET].dropna().unique().tolist()) == [0, 1]
        flag = OK if binaire else KO
        print(
            f"  {flag} cible '{TARGET}' : {n_classes} valeurs uniques "
            f"{'(binaire 0/1)' if binaire else '(NON binaire !)'}"
        )
        if not binaire:
            errors.append(f"cible '{TARGET}' non binaire")

    manquantes = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
    if manquantes:
        errors.append(f"colonnes introuvables : {manquantes}")
        print(f"  {KO} colonnes introuvables dans le CSV : {manquantes}")
    else:
        print(
            f"  {OK} {len(NUMERIC_FEATURES)} num + {len(CATEGORICAL_FEATURES)} cat "
            f"toutes presentes dans le CSV"
        )

    if errors:
        print(f"\n{KO} config.py invalide, on s'arrete avant data.py/features.py.")
        for e in errors:
            print(f"   - {e}")
        return 1

    # --- 3. data.py : split ------------------------------------------------
    x_train, x_test, y_train, y_test = split(df)
    print(f"\n[3] data.py  -> split : train={len(x_train):,}  test={len(x_test):,}")
    tr = y_train.mean() * 100
    te = y_test.mean() * 100
    print(f"  {OK} taux de fraude train={tr:.2f}%  test={te:.2f}%  (stratifie)")

    # --- 4. features.py : pre-processing -----------------------------------
    pre = build_preprocessor()
    x_train_t = pre.fit_transform(x_train)
    x_test_t = pre.transform(x_test)
    n_in = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    n_out = x_train_t.shape[1]
    print("\n[4] features.py -> pre-processing applique")
    print(
        f"  {OK} colonnes : {n_in} en entree -> {n_out} apres encodage "
        f"(OneHot sur {len(CATEGORICAL_FEATURES)} colonnes cat.)"
    )
    print(f"  {OK} train transforme : {x_train_t.shape}   test : {x_test_t.shape}")

    print("\n" + "=" * 70)
    print(f"{OK} Les 3 fichiers sont valides : la chaine est prete pour l'entrainement.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
