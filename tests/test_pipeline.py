from __future__ import annotations

import numpy as np
import pandas as pd

from mlproject.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from mlproject.data import split
from mlproject.features import build_preprocessor


def _synthetic(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data: dict = {c: rng.normal(size=n) for c in NUMERIC_FEATURES}
    for c in CATEGORICAL_FEATURES:
        data[c] = rng.choice(["A", "B", "C"], size=n)
    data[TARGET] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data)


def test_config_not_empty() -> None:
    assert TARGET
    assert len(NUMERIC_FEATURES) > 0
    assert len(CATEGORICAL_FEATURES) > 0


def test_preprocessor_transforms() -> None:
    df = _synthetic()
    pre = build_preprocessor()
    out = pre.fit_transform(df)
    assert out.shape[0] == len(df)
    assert out.shape[1] >= len(NUMERIC_FEATURES)


def test_split_keeps_all_rows() -> None:
    df = _synthetic()
    x_train, x_test, y_train, y_test = split(df)
    assert len(x_train) + len(x_test) == len(df)
    assert TARGET not in x_train.columns
