from typing import Any

import pandas as pd

from backend.model.predict import predict_with_confidence


def _risk_level_from_probability(probability: float) -> str:
    if probability >= 0.75:
        return "high"
    if probability >= 0.45:
        return "medium"
    return "low"


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.65:
        return "medium"
    return "low"


def _top_feature_contributions(
    features: dict[str, Any] | pd.Series,
    prediction_bundle: dict[str, Any],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    if isinstance(features, pd.Series):
        feature_map = features.to_dict()
    else:
        feature_map = dict(features)

    model_meta = prediction_bundle.get("model_metadata", {})
    best_model_name = model_meta.get("best_model_name")
    model_block = model_meta.get("models", {}).get(best_model_name, {})
    importance_block = model_block.get("feature_importance", {})
    ranked = importance_block.get("top_features", [])

    if not ranked:
        return []

    contributions = []
    for row in ranked:
        name = row.get("feature")
        if name is None:
            continue
        value = feature_map.get(name)
        contributions.append(
            {
                "feature": name,
                "importance": float(row.get("importance", 0.0)),
                "student_value": value,
            }
        )
        if len(contributions) >= top_n:
            break
    return contributions


def generate_risk_profile(
    student_id: str | int,
    features: dict[str, Any] | pd.Series,
    prediction_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Create personalized counseling profile."""
    model = prediction_bundle["model"]
    scaler = prediction_bundle.get("scaler")
    feature_names = prediction_bundle.get("feature_names")
    if scaler is None or feature_names is None:
        raise ValueError("prediction_bundle must include scaler and feature_names.")

    X_new = pd.DataFrame([dict(features)])
    pred = predict_with_confidence(
        X_new,
        model=model,
        scaler=scaler,
        feature_names=feature_names,
    )[0]

    dropout_probability = float(pred["dropout_probability"])
    confidence = float(pred["confidence"])
    risk_level = _risk_level_from_probability(dropout_probability)

    contributing_factors = _top_feature_contributions(features, prediction_bundle)
    profile = {
        "student_id": student_id,
        "risk_level": risk_level,
        "dropout_probability": dropout_probability,
        "prediction": int(pred["prediction"]),
        "confidence_score": confidence,
        "confidence_level": _confidence_label(confidence),
        "contributing_factors": contributing_factors,
        "recommendations": [],
    }
    profile["recommendations"] = recommend_interventions(profile)
    return profile


def recommend_interventions(risk_profile: dict[str, Any]) -> list[dict[str, str]]:
    """Generate counseling recommendations."""
    factors = risk_profile.get("contributing_factors", [])
    factor_names = [str(row.get("feature", "")).lower() for row in factors]
    risk_level = str(risk_profile.get("risk_level", "low")).lower()

    recommendations: list[dict[str, str]] = []

    def add(category: str, action: str, priority: str) -> None:
        recommendations.append(
            {
                "category": category,
                "action": action,
                "priority": priority,
            }
        )

    if any(k in " ".join(factor_names) for k in ["grade", "gpa", "attendance", "backlog", "score"]):
        add(
            "academic_support",
            "Schedule weekly tutoring, attendance monitoring, and a faculty check-in plan.",
            "high" if risk_level == "high" else "medium",
        )

    if any(k in " ".join(factor_names) for k in ["income", "financial", "fee", "scholarship"]):
        add(
            "financial_support",
            "Connect student with scholarships, emergency grants, and fee-payment counseling.",
            "high" if risk_level in {"high", "medium"} else "medium",
        )

    if any(k in " ".join(factor_names) for k in ["stress", "engagement", "mental", "wellbeing", "depression"]):
        add(
            "mental_health",
            "Arrange counselor sessions and monthly wellbeing follow-ups with consent.",
            "high" if risk_level == "high" else "medium",
        )

    add(
        "mentorship_program",
        "Assign a senior mentor and define fortnightly progress reviews.",
        "high" if risk_level == "high" else "medium",
    )

    if risk_level == "high":
        add(
            "retention_case_management",
            "Create a 30-day intervention plan with academic, financial, and counseling checkpoints.",
            "high",
        )
    elif risk_level == "medium":
        add(
            "preventive_monitoring",
            "Track attendance/performance biweekly and escalate if risk indicators worsen.",
            "medium",
        )
    else:
        add(
            "light_touch_followup",
            "Maintain monthly check-ins and encourage participation in engagement activities.",
            "low",
        )

    return recommendations

