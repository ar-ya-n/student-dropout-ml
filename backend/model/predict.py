import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.preprocessing.preprocess import load_preprocessing_artifacts


def _resolve_artifact_paths(artifacts_dir: str | Path) -> dict[str, Path]:
    root = Path(artifacts_dir)
    if not root.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {root}")

    # Support either a flat artifacts directory or separate subfolders.
    model_dir = root / "model" if (root / "model").is_dir() else root
    preprocess_dir = (
        root / "preprocessing" if (root / "preprocessing").is_dir() else root
    )

    return {"root": root, "model_dir": model_dir, "preprocess_dir": preprocess_dir}


def load_best_model(artifacts_dir: str | Path) -> dict[str, Any]:
    """Load best model and preprocessing artifacts."""
    paths = _resolve_artifact_paths(artifacts_dir)
    model_path = paths["model_dir"] / "best_model.pkl"
    metadata_path = paths["model_dir"] / "model_metadata.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Best model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    model_metadata = {}
    if metadata_path.is_file():
        with open(metadata_path, encoding="utf-8") as f:
            model_metadata = json.load(f)

    preprocess_bundle = load_preprocessing_artifacts(paths["preprocess_dir"])
    return {
        "model": model,
        "scaler": preprocess_bundle["scaler"],
        "feature_names": preprocess_bundle["selected_features"],
        "preprocessing_metadata": preprocess_bundle["metadata"],
        "model_metadata": model_metadata,
    }


def _prepare_features(
    X_new: pd.DataFrame | np.ndarray | list[list[float]],
    feature_names: list[str],
) -> pd.DataFrame:
    if isinstance(X_new, pd.DataFrame):
        df = X_new.copy()
    else:
        arr = np.asarray(X_new)
        if arr.ndim != 2:
            raise ValueError("X_new must be 2D for batch prediction.")
        if arr.shape[1] == len(feature_names):
            df = pd.DataFrame(arr, columns=feature_names)
        else:
            df = pd.DataFrame(arr)

    for col in ("Student_ID", "Dropout"):
        if col in df.columns:
            df = df.drop(columns=[col])

    # One-hot encode incoming categorical inputs and align to training features.
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    df = df.reindex(columns=feature_names, fill_value=0)
    return df


def predict_batch(
    X_new: pd.DataFrame | np.ndarray | list[list[float]],
    model: Any,
    scaler: Any,
    feature_names: list[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Batch predictions with preprocessing.

    Parameters
    ----------
    threshold : float, default=0.5
        Classification threshold for probability predictions.
        Use optimized threshold (e.g., 0.45) for better F1-Score.
    """
    X_aligned = _prepare_features(X_new, feature_names)
    X_scaled = scaler.transform(X_aligned)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[:, 1]
        preds = (proba >= threshold).astype(int)
    else:
        preds = model.predict(X_scaled)

    output = pd.DataFrame({"prediction": preds.astype(int)})

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[:, 1]
        output["dropout_probability"] = proba
    return output


def predict_with_confidence(
    X_new: pd.DataFrame | np.ndarray | list[list[float]],
    model: Any,
    threshold: float = 0.5,
    scaler: Any = None,
    feature_names: list[str] | None = None,
) -> list[dict[str, float | int]]:
    """Individual predictions with confidence scores."""
    if scaler is not None and feature_names is not None:
        X_aligned = _prepare_features(X_new, feature_names)
        X_new = scaler.transform(X_aligned)

    if not hasattr(model, "predict_proba"):
        preds = model.predict(X_new)
        return [
            {"prediction": int(pred), "dropout_probability": float("nan"), "confidence": float("nan")}
            for pred in preds
        ]

    probs = model.predict_proba(X_new)[:, 1]
    results: list[dict[str, float | int]] = []
    for p in probs:
        pred = int(p >= threshold)
        confidence = float(p if pred == 1 else 1.0 - p)
        results.append(
            {
                "prediction": pred,
                "dropout_probability": float(p),
                "confidence": confidence,
            }
        )
    return results
