import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str
    estimator: Any
    param_distributions: dict[str, list[Any]]
    n_iter: int = 20


def _ensure_logging_configured() -> None:
    if not logging.getLogger().handlers and not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def _safe_import_xgboost():
    try:
        from xgboost import XGBClassifier  # type: ignore[reportMissingImports]

        return XGBClassifier
    except Exception as exc:
        logger.warning("xgboost not available, skipping XGBoost model: %s", exc)
        return None


def _safe_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier  # type: ignore[reportMissingImports]

        return LGBMClassifier
    except Exception as exc:
        logger.warning("lightgbm not available, skipping LightGBM model: %s", exc)
        return None


def create_default_model_configs(random_state: int = 42) -> list[ModelConfig]:
    _ensure_logging_configured()
    configs: list[ModelConfig] = []

    configs.append(
        ModelConfig(
            name="logistic_regression",
            estimator=LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                solver="lbfgs",
                random_state=random_state,
            ),
            param_distributions={
                "C": np.logspace(-3, 2, 20).tolist(),
            },
            n_iter=15,
        )
    )

    configs.append(
        ModelConfig(
            name="random_forest",
            estimator=RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            param_distributions={
                "n_estimators": [200, 300, 500, 700],
                "max_depth": [None, 5, 10, 15, 20, 30],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2", None],
            },
            n_iter=20,
        )
    )

    xgb_cls = _safe_import_xgboost()
    if xgb_cls is not None:
        configs.append(
            ModelConfig(
                name="xgboost",
                estimator=xgb_cls(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_estimators=400,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                ),
                param_distributions={
                    "n_estimators": [200, 400, 600],
                    "learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "max_depth": [3, 4, 5, 6, 8],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
                },
                n_iter=20,
            )
        )

    lgbm_cls = _safe_import_lightgbm()
    if lgbm_cls is not None:
        configs.append(
            ModelConfig(
                name="lightgbm",
                estimator=lgbm_cls(
                    objective="binary",
                    random_state=random_state,
                    n_estimators=500,
                    learning_rate=0.05,
                ),
                param_distributions={
                    "n_estimators": [200, 400, 600],
                    "learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "num_leaves": [15, 31, 63, 127],
                    "max_depth": [-1, 5, 10, 15],
                    "subsample": [0.7, 0.8, 0.9, 1.0],
                    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                },
                n_iter=20,
            )
        )

    configs.append(
        ModelConfig(
            name="neural_network",
            estimator=MLPClassifier(
                random_state=random_state,
                max_iter=400,
                early_stopping=True,
                n_iter_no_change=15,
            ),
            param_distributions={
                "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
                "activation": ["relu", "tanh"],
                "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3],
                "batch_size": [32, 64, 128],
            },
            n_iter=15,
        )
    )

    return configs


def _evaluate_binary_model(model: Any, X_test: Any, y_test: Any) -> dict[str, float]:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _build_confusion_analysis(y_true: Any, y_pred: Any) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    return {
        "labels": [0, 1],
        "matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def _extract_feature_importance(
    model: Any,
    X_test: Any,
    y_test: Any,
    feature_names: Optional[list[str]],
    *,
    random_state: int,
    top_n: int = 20,
) -> dict[str, Any]:
    names = feature_names or [f"feature_{i}" for i in range(np.asarray(X_test).shape[1])]
    importances: Optional[np.ndarray] = None
    method = "none"

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        method = "feature_importances_"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef[0] if coef.ndim > 1 else coef)
        method = "abs_coef"
    else:
        # Fallback for models without native importance (e.g., MLP)
        try:
            pi = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=10,
                random_state=random_state,
                scoring="f1",
            )
            importances = np.asarray(pi.importances_mean)
            method = "permutation_importance"
        except Exception:
            return {"method": "unavailable", "top_features": []}

    if importances is None or importances.shape[0] != len(names):
        return {"method": "unavailable", "top_features": []}

    ranking = np.argsort(importances)[::-1]
    top = []
    for idx in ranking[:top_n]:
        top.append({"feature": str(names[idx]), "importance": float(importances[idx])})
    return {"method": method, "top_features": top}


def train_and_tune_models(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    *,
    model_names: Optional[list[str]] = None,
    random_state: int = 42,
    cv_folds: int = 5,
    scoring: str = "f1",
    n_jobs: int = -1,
    verbose: int = 0,
    feature_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Train baseline + advanced classifiers with cross-validated hyperparameter tuning.

    model_names can include:
    - logistic_regression
    - random_forest
    - xgboost
    - lightgbm
    - neural_network
    """
    _ensure_logging_configured()
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)
    classes = np.unique(y_train_arr)
    if len(classes) != 2:
        raise ValueError(f"Expected binary target for dropout, found classes: {classes.tolist()}")

    configs = create_default_model_configs(random_state=random_state)
    if model_names:
        wanted = set(model_names)
        configs = [c for c in configs if c.name in wanted]
        if not configs:
            raise ValueError(f"No valid models selected. Received model_names={model_names}")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    results: dict[str, Any] = {
        "models": {},
        "leaderboard": [],
        "best_model_name": None,
        "best_estimator": None,
    }

    for cfg in configs:
        logger.info("Tuning model: %s", cfg.name)
        search = RandomizedSearchCV(
            estimator=cfg.estimator,
            param_distributions=cfg.param_distributions,
            n_iter=cfg.n_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv,
            random_state=random_state,
            verbose=verbose,
            refit=True,
        )
        search.fit(X_train, y_train_arr)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        test_metrics = _evaluate_binary_model(best_model, X_test, y_test_arr)
        confusion = _build_confusion_analysis(y_test_arr, y_pred)
        feature_importance = _extract_feature_importance(
            best_model,
            X_test,
            y_test_arr,
            feature_names=feature_names,
            random_state=random_state,
        )
        result_row = {
            "name": cfg.name,
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "test_metrics": test_metrics,
            "confusion_matrix": confusion,
            "feature_importance": feature_importance,
            "estimator": best_model,
        }
        results["models"][cfg.name] = result_row
        results["leaderboard"].append(
            {
                "name": cfg.name,
                "best_cv_score": float(search.best_score_),
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
            }
        )

    results["leaderboard"] = sorted(
        results["leaderboard"],
        key=lambda row: (row["test_f1"], row["best_cv_score"]),
        reverse=True,
    )
    best_name = results["leaderboard"][0]["name"]
    results["best_model_name"] = best_name
    results["best_estimator"] = results["models"][best_name]["estimator"]
    return results


def save_model_artifacts(
    output_dir: str | Path,
    train_results: dict[str, Any],
    *,
    include_all_models: bool = False,
) -> dict[str, Path]:
    """
    Save best estimator and leaderboard/metadata for inference and tracking.
    """
    _ensure_logging_configured()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    best_model_path = out / "best_model.pkl"
    with open(best_model_path, "wb") as f:
        pickle.dump(train_results["best_estimator"], f)
    files["best_model"] = best_model_path

    leaderboard_path = out / "leaderboard.json"
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump(train_results["leaderboard"], f, indent=2)
    files["leaderboard"] = leaderboard_path

    metadata_path = out / "model_metadata.json"
    metadata = {
        "best_model_name": train_results["best_model_name"],
        "models": {},
    }
    for name, row in train_results["models"].items():
        metadata["models"][name] = {
            "best_params": row["best_params"],
            "best_cv_score": row["best_cv_score"],
            "test_metrics": row["test_metrics"],
            "confusion_matrix": row["confusion_matrix"],
            "feature_importance": row["feature_importance"],
        }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    files["metadata"] = metadata_path

    if include_all_models:
        all_models_dir = out / "all_models"
        all_models_dir.mkdir(parents=True, exist_ok=True)
        for name, row in train_results["models"].items():
            model_path = all_models_dir / f"{name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(row["estimator"], f)
        files["all_models"] = all_models_dir

    logger.info("Saved model artifacts under %s", out.resolve())
    return files

