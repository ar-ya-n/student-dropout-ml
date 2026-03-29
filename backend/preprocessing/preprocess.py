import json
import logging
import pickle
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


logger = logging.getLogger(__name__)

TARGET_COLUMN = "Dropout"
ID_COLUMNS = ("Student_ID",)

DEFAULT_CORRELATION_THRESHOLD = 0.12
DEFAULT_MUTUAL_INFO_THRESHOLD = 0.01

FeatureSelectionMethod = Literal["correlation", "mutual_info"]
BalanceStrategy = Literal["smote", "random_oversample", None]


def _ensure_logging_configured() -> None:
    if not logging.getLogger().handlers and not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def validate_dataframe(
    df: pd.DataFrame,
    *,
    required_columns: Optional[list[str]] = None,
    allow_missing: bool = False,
    outlier_check: Literal["none", "warn"] = "warn",
    outlier_iqr_factor: float = 3.0,
) -> None:
    """
    Validate inputs: required columns, missing values, rough outlier flags, dtype sanity.
    Raises ValueError on hard failures; logs warnings for outliers when outlier_check='warn'.
    """
    _ensure_logging_configured()
    req = required_columns if required_columns is not None else [TARGET_COLUMN]
    cols = [c.strip() for c in df.columns]
    missing_req = [c for c in req if c not in cols]
    if missing_req:
        raise ValueError(
            f"Missing required column(s): {missing_req}. Present columns: {cols}"
        )

    if TARGET_COLUMN in df.columns and df[TARGET_COLUMN].dtype == "object":
        logger.warning(
            "Target column %r is object dtype; expected numeric/binary labels.",
            TARGET_COLUMN,
        )

    null_cols = df.columns[df.isna().any()].tolist()
    if null_cols and not allow_missing:
        raise ValueError(
            f"Missing values found in column(s): {null_cols}. "
            "Handle or impute before preprocessing, or pass allow_missing=True."
        )

    if outlier_check == "warn":
        num_cols = df.select_dtypes(include=[np.number]).columns.drop(
            TARGET_COLUMN, errors="ignore"
        )
        for col in num_cols:
            s = df[col].dropna()
            if len(s) < 10:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            low, high = q1 - outlier_iqr_factor * iqr, q3 + outlier_iqr_factor * iqr
            n_out = ((s < low) | (s > high)).sum()
            if n_out > 0:
                logger.warning(
                    "Column %r: %s values beyond %.1f*IQR (approximate outliers).",
                    col,
                    int(n_out),
                    outlier_iqr_factor,
                )


def load_data(
    path: Union[str, Path],
    *,
    required_columns: Optional[list[str]] = None,
    validate: bool = True,
    **validate_kwargs: Any,
) -> pd.DataFrame:
    """Load CSV with optional validation. Raises FileNotFoundError if path is missing."""
    _ensure_logging_configured()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path.resolve()}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {path}") from e

    df.columns = df.columns.str.strip()
    req = required_columns if required_columns is not None else [TARGET_COLUMN]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required column(s): {missing}. Found: {list(df.columns)}"
        )

    if validate:
        validate_dataframe(df, required_columns=req, **validate_kwargs)

    logger.info("Loaded %s rows, %s columns from %s", len(df), len(df.columns), path)
    return df


def _is_categorical_like(series: pd.Series) -> bool:
    dt = series.dtype
    if isinstance(dt, CategoricalDtype):
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series):
        return True
    if pd.api.types.is_string_dtype(series):
        return True
    return False


def _encode_categorical(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    categorical_cols = [c for c in train_df.columns if _is_categorical_like(train_df[c])]
    train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=False)
    test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=False)
    train_encoded, test_encoded = train_encoded.align(test_encoded, join="left", axis=1, fill_value=0)
    return train_encoded, test_encoded


def _select_features_by_correlation(
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    threshold: float,
) -> Tuple[list[str], pd.Series]:
    train_with_target = X_train_df.copy()
    train_with_target[TARGET_COLUMN] = y_train.values

    corr_to_target = train_with_target.corr(numeric_only=True)[TARGET_COLUMN].drop(
        labels=[TARGET_COLUMN], errors="ignore"
    )
    corr_to_target = corr_to_target.abs().sort_values(ascending=False)

    selected_features = corr_to_target[corr_to_target >= threshold].index.tolist()
    if not selected_features:
        logger.warning(
            "No features met correlation threshold %s; using all %s features.",
            threshold,
            len(X_train_df.columns),
        )
        selected_features = X_train_df.columns.tolist()

    return selected_features, corr_to_target


def _select_features_by_mutual_info(
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int,
    threshold: float,
    top_k: Optional[int] = None,
) -> Tuple[list[str], pd.Series]:
    X = X_train_df.values
    y = y_train.values
    mi = mutual_info_classif(X, y, random_state=random_state)
    scores = pd.Series(mi, index=X_train_df.columns).sort_values(ascending=False)

    if top_k is not None:
        selected_features = scores.head(top_k).index.tolist()
    else:
        selected_features = scores[scores >= threshold].index.tolist()

    if not selected_features:
        logger.warning(
            "No features met mutual information threshold %s; using all features.",
            threshold,
        )
        selected_features = X_train_df.columns.tolist()

    return selected_features, scores


def _balance_training_data(
    X_train_df: pd.DataFrame,
    y_train: pd.Series,
    *,
    strategy: BalanceStrategy,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    if strategy is None:
        return X_train_df, y_train

    class_counts = y_train.value_counts()
    if len(class_counts) < 2 or class_counts.min() == class_counts.max():
        return X_train_df, y_train

    minority_count = int(class_counts.min())

    if strategy == "smote":
        k_neighbors = max(1, min(5, minority_count - 1))
        try:
            smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
            X_res, y_res = smote.fit_resample(X_train_df, y_train)
            if isinstance(X_res, np.ndarray):
                X_balanced = pd.DataFrame(X_res, columns=X_train_df.columns)
            else:
                X_balanced = X_res
            y_balanced = pd.Series(y_res, name=TARGET_COLUMN)
            logger.info("Applied SMOTE (k_neighbors=%s).", k_neighbors)
            return X_balanced, y_balanced
        except Exception as e:
            logger.warning("SMOTE failed (%s); falling back to random oversampling.", e)

    train_df = X_train_df.copy()
    train_df[TARGET_COLUMN] = y_train.values
    class_counts = train_df[TARGET_COLUMN].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    majority_df = train_df[train_df[TARGET_COLUMN] == majority_class]
    minority_df = train_df[train_df[TARGET_COLUMN] == minority_class]

    minority_upsampled = resample(
        minority_df,
        replace=True,
        n_samples=len(majority_df),
        random_state=random_state,
    )
    balanced_df = pd.concat([majority_df, minority_upsampled], axis=0).sample(
        frac=1, random_state=random_state
    )
    X_balanced = balanced_df.drop(columns=[TARGET_COLUMN])
    y_balanced = balanced_df[TARGET_COLUMN]
    logger.info("Applied random oversampling to balance classes.")
    return X_balanced, y_balanced


def save_preprocessing_artifacts(
    directory: Union[str, Path],
    scaler: StandardScaler,
    selected_features: list[str],
    metadata: dict[str, Any],
    *,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Path]:
    """
    Persist scaler, feature list, and metadata for inference.
    Returns paths to written files.
    """
    _ensure_logging_configured()
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    scaler_path = out / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    paths["scaler"] = scaler_path

    features_path = out / "selected_features.json"
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(selected_features, f, indent=2)
    paths["selected_features"] = features_path

    meta_path = out / "preprocessing_metadata.json"
    serializable = {k: v for k, v in metadata.items() if k != "correlation_scores" and k != "mutual_info_scores"}
    if "correlation_scores" in metadata:
        cs = metadata["correlation_scores"]
        serializable["correlation_scores"] = (
            {str(k): float(v) for k, v in cs.items()} if isinstance(cs, dict) else str(cs)
        )
    if "mutual_info_scores" in metadata:
        ms = metadata["mutual_info_scores"]
        serializable["mutual_info_scores"] = (
            {str(k): float(v) for k, v in ms.items()} if isinstance(ms, dict) else str(ms)
        )
    def _json_default(o: Any) -> Any:
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=_json_default)
    paths["metadata"] = meta_path

    if extra:
        extra_path = out / "extra.json"
        with open(extra_path, "w", encoding="utf-8") as f:
            json.dump(extra, f, indent=2, default=str)
        paths["extra"] = extra_path

    logger.info("Saved preprocessing artifacts under %s", out.resolve())
    return paths


def load_preprocessing_artifacts(
    directory: Union[str, Path],
) -> dict[str, Any]:
    """Load scaler, feature list, and metadata saved by ``save_preprocessing_artifacts``."""
    _ensure_logging_configured()
    out = Path(directory)
    if not out.is_dir():
        raise FileNotFoundError(f"Artifacts directory not found: {out.resolve()}")

    scaler_path = out / "scaler.pkl"
    features_path = out / "selected_features.json"
    meta_path = out / "preprocessing_metadata.json"

    for p in (scaler_path, features_path, meta_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing artifact file: {p}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(features_path, encoding="utf-8") as f:
        selected_features = json.load(f)
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info("Loaded preprocessing artifacts from %s", out.resolve())
    return {
        "scaler": scaler,
        "selected_features": selected_features,
        "metadata": metadata,
    }


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    *,
    feature_selection_method: FeatureSelectionMethod = "correlation",
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    mutual_info_threshold: float = DEFAULT_MUTUAL_INFO_THRESHOLD,
    mutual_info_top_k: Optional[int] = None,
    balance_strategy: BalanceStrategy = "smote",
    validate_input: bool = True,
    allow_missing: bool = False,
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    pd.Series,
    pd.Series,
    StandardScaler,
    dict[str, Any],
]:
    """
    Full preprocessing: optional validation, split, encode, feature selection,
    imbalance handling (SMOTE by default), scaling.

    If ``artifacts_dir`` is set, writes scaler, selected_features.json, and metadata JSON.
    """
    _ensure_logging_configured()

    df = df.copy()
    columns_to_drop = [col for col in ID_COLUMNS if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    if validate_input:
        validate_dataframe(df, allow_missing=allow_missing)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"DataFrame must contain target column {TARGET_COLUMN!r}.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    X_train_encoded, X_test_encoded = _encode_categorical(X_train, X_test)

    correlation_scores: Optional[pd.Series] = None
    mutual_info_scores: Optional[pd.Series] = None

    if feature_selection_method == "correlation":
        selected_features, correlation_scores = _select_features_by_correlation(
            X_train_encoded, y_train, threshold=correlation_threshold
        )
    elif feature_selection_method == "mutual_info":
        selected_features, mutual_info_scores = _select_features_by_mutual_info(
            X_train_encoded,
            y_train,
            random_state=random_state,
            threshold=mutual_info_threshold,
            top_k=mutual_info_top_k,
        )
    else:
        raise ValueError(f"Unknown feature_selection_method: {feature_selection_method}")

    X_train_selected = X_train_encoded[selected_features]
    X_test_selected = X_test_encoded[selected_features]

    X_train_balanced, y_train_balanced = _balance_training_data(
        X_train_selected,
        y_train,
        strategy=balance_strategy,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_selected)

    metadata: dict[str, Any] = {
        "selected_features": selected_features,
        "class_distribution_before": y_train.value_counts(normalize=True).to_dict(),
        "class_distribution_after": y_train_balanced.value_counts(normalize=True).to_dict(),
        "test_size": test_size,
        "stratified": stratify,
        "random_state": random_state,
        "feature_selection_method": feature_selection_method,
        "correlation_threshold": correlation_threshold,
        "mutual_info_threshold": mutual_info_threshold,
        "mutual_info_top_k": mutual_info_top_k,
        "balance_strategy": balance_strategy,
    }
    if correlation_scores is not None:
        metadata["correlation_scores"] = correlation_scores.to_dict()
    if mutual_info_scores is not None:
        metadata["mutual_info_scores"] = mutual_info_scores.to_dict()

    if artifacts_dir is not None:
        save_preprocessing_artifacts(
            artifacts_dir,
            scaler,
            selected_features,
            metadata,
            extra={"target_column": TARGET_COLUMN, "id_columns": list(ID_COLUMNS)},
        )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_balanced,
        y_test,
        scaler,
        metadata,
    )
