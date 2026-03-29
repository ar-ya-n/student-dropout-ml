"""
Flask API for dropout prediction and counseling recommendations.

Set ARTIFACTS_DIR to the folder containing model + preprocessing artifacts
(e.g. parent of `model/` and `preprocessing/`, or a flat folder with both).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request

from backend.counseling.recommendation import generate_risk_profile
from backend.model.predict import load_best_model, predict_batch

logger = logging.getLogger(__name__)

app = Flask(__name__)

_prediction_bundle: dict[str, Any] | None = None
_profile_cache: dict[str, dict[str, Any]] = {}


def get_prediction_bundle() -> dict[str, Any]:
    """Lazy-load model + preprocessing artifacts (cached)."""
    global _prediction_bundle
    if _prediction_bundle is None:
        artifacts_dir = os.environ.get("ARTIFACTS_DIR", "artifacts")
        _prediction_bundle = load_best_model(artifacts_dir)
        logger.info("Loaded prediction bundle from %s", artifacts_dir)
    return _prediction_bundle


def _cache_profile(student_id: str, profile: dict[str, Any]) -> None:
    _profile_cache[str(student_id)] = profile


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict/single")
def predict_single_student():
    """
    Predict for one student with counseling recommendations.

    JSON body (preferred):
      { "student_id": "<id>", "features": { ... feature columns ... } }

    Alternative: flat body with student_id / Student_ID plus feature keys.
    """
    try:
        bundle = get_prediction_bundle()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    data = request.get_json(silent=True) or {}
    student_id = data.get("student_id") or data.get("Student_ID")
    features = data.get("features")
    if features is None:
        features = {
            k: v
            for k, v in data.items()
            if k not in ("student_id", "Student_ID", "features")
        }
    if not features:
        return jsonify(
            {
                "error": "Missing features. Send JSON with 'features' object or flat feature keys.",
            }
        ), 400
    if student_id is None:
        student_id = "anonymous"

    profile = generate_risk_profile(student_id, features, bundle)
    _cache_profile(str(student_id), profile)

    return jsonify(
        {
            "student_id": profile["student_id"],
            "prediction": profile["prediction"],
            "dropout_probability": profile["dropout_probability"],
            "risk_level": profile["risk_level"],
            "confidence_score": profile["confidence_score"],
            "confidence_level": profile["confidence_level"],
            "contributing_factors": profile["contributing_factors"],
            "recommendations": profile["recommendations"],
        }
    )


@app.post("/predict/batch")
def predict_batch_upload():
    """
    Batch predictions from CSV upload.

    Multipart form field: `file` (CSV). Optional: `include_counseling=true`
    to attach counseling per row when `Student_ID` (or `student_id`) column exists.
    """
    try:
        bundle = get_prediction_bundle()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    if "file" not in request.files:
        return jsonify({"error": "Missing file field 'file' (CSV upload)."}), 400

    upload = request.files["file"]
    if not upload or upload.filename == "":
        return jsonify({"error": "Empty file upload."}), 400

    try:
        df = pd.read_csv(upload)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    df.columns = df.columns.str.strip()
    include_counseling = request.args.get("include_counseling", "false").lower() in (
        "1",
        "true",
        "yes",
    )

    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    out_df = predict_batch(df, model, scaler, feature_names)
    result_df = pd.concat([df.reset_index(drop=True), out_df.reset_index(drop=True)], axis=1)

    id_col = None
    for cand in ("Student_ID", "student_id"):
        if cand in result_df.columns:
            id_col = cand
            break

    counseling_profiles: list[dict[str, Any]] | None = None
    if include_counseling and id_col is not None:
        counseling_profiles = []
        for _, row in result_df.iterrows():
            sid = row[id_col]
            drop_labels = list(out_df.columns) + [id_col]
            feat_row = row.drop(labels=drop_labels, errors="ignore")
            for extra in ("Dropout", "dropout"):
                if extra in feat_row.index:
                    feat_row = feat_row.drop(labels=[extra], errors="ignore")
            features = feat_row.to_dict()
            prof = generate_risk_profile(sid, features, bundle)
            _cache_profile(str(sid), prof)
            counseling_profiles.append(prof)

    response: dict[str, Any] = {
        "count": len(result_df),
        "predictions": result_df.to_dict(orient="records"),
    }
    if counseling_profiles is not None:
        response["counseling_profiles"] = counseling_profiles

    return jsonify(response)


@app.get("/student/<student_id>/counseling-plan")
def get_counseling_plan(student_id: str):
    """
    Return personalized counseling recommendations for a student.

    Profiles are populated after POST /predict/single or batch with
    include_counseling=true (and a Student_ID column). Otherwise returns 404.
    """
    key = str(student_id)
    if key not in _profile_cache:
        return jsonify(
            {
                "error": "No counseling profile found for this student.",
                "hint": "Call POST /predict/single with this student_id first, or "
                "POST /predict/batch?include_counseling=true with a Student_ID column.",
            }
        ), 404
    return jsonify(_profile_cache[key])


def create_app() -> Flask:
    """Factory for WSGI servers (gunicorn, uwsgi)."""
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False)
