import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.preprocessing.preprocess import (
    TARGET_COLUMN,
    load_data,
    load_preprocessing_artifacts,
    preprocess_data,
    save_preprocessing_artifacts,
    validate_dataframe,
)


def _synthetic_imbalanced_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_minority = max(30, int(n * 0.24))
    n_majority = n - n_minority
    y = np.concatenate([np.ones(n_minority), np.zeros(n_majority)])
    rng.shuffle(y)
    score = rng.normal(65.0, 8.0, n) + 5.0 * y
    income = rng.choice(["Low", "Mid", "High"], n)
    dept = rng.choice(["CS", "EE"], n)
    return pd.DataFrame(
        {
            "Student_ID": range(n),
            "Score": score,
            "Income_Bracket": income,
            "Department": dept,
            TARGET_COLUMN: y.astype(int),
        }
    )


def test_validate_raises_on_missing_values():
    df = _synthetic_imbalanced_df(50)
    df.loc[0, "Score"] = np.nan
    with pytest.raises(ValueError, match="Missing values"):
        validate_dataframe(df, allow_missing=False)


def test_validate_passes_with_allow_missing():
    df = _synthetic_imbalanced_df(50)
    df.loc[0, "Score"] = np.nan
    validate_dataframe(df, allow_missing=True)


def test_load_data_file_not_found(tmp_path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        load_data(missing)


def test_load_data_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_data(p)


def test_preprocess_smoke_correlation_smote():
    df = _synthetic_imbalanced_df(200)
    X_train, X_test, y_train, y_test, scaler, meta = preprocess_data(
        df,
        random_state=7,
        feature_selection_method="correlation",
        correlation_threshold=0.05,
        balance_strategy="smote",
        validate_input=True,
    )
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert abs(y_train.value_counts(normalize=True).iloc[0] - 0.5) < 0.01
    assert meta["balance_strategy"] == "smote"
    assert meta["random_state"] == 7
    assert "correlation_scores" in meta


def test_preprocess_mutual_info():
    df = _synthetic_imbalanced_df(200)
    _, _, _, _, _, meta = preprocess_data(
        df,
        feature_selection_method="mutual_info",
        mutual_info_threshold=0.0,
        mutual_info_top_k=5,
        balance_strategy="smote",
    )
    assert len(meta["selected_features"]) <= 5
    assert "mutual_info_scores" in meta


def test_save_artifacts_writes_files(tmp_path):
    df = _synthetic_imbalanced_df(120)
    _, _, _, _, scaler, meta = preprocess_data(
        df,
        artifacts_dir=tmp_path,
        correlation_threshold=0.01,
    )
    assert (tmp_path / "scaler.pkl").is_file()
    assert (tmp_path / "selected_features.json").is_file()
    assert (tmp_path / "preprocessing_metadata.json").is_file()
    with open(tmp_path / "selected_features.json", encoding="utf-8") as f:
        features = json.load(f)
    assert isinstance(features, list)
    with open(tmp_path / "scaler.pkl", "rb") as f:
        loaded = pickle.load(f)
    assert type(loaded).__name__ == type(scaler).__name__

    loaded_bundle = load_preprocessing_artifacts(tmp_path)
    assert "scaler" in loaded_bundle and "selected_features" in loaded_bundle
    assert loaded_bundle["selected_features"] == meta["selected_features"]


def test_save_preprocessing_artifacts_standalone(tmp_path):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    meta = {"selected_features": ["a"], "balance_strategy": None}
    paths = save_preprocessing_artifacts(
        tmp_path, scaler, ["a", "b"], meta, extra={"k": 1}
    )
    assert "scaler" in paths


def test_random_oversample_fallback_path():
    df = _synthetic_imbalanced_df(80)
    _, _, y_train, _, _, meta = preprocess_data(
        df,
        balance_strategy="random_oversample",
        correlation_threshold=0.01,
    )
    assert meta["balance_strategy"] == "random_oversample"
    assert abs(y_train.value_counts(normalize=True).iloc[0] - 0.5) < 0.01
