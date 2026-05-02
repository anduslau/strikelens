from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.feature_engineering import build_features_for_event_window


def _load_model_bundle(model_path: str | Path) -> tuple[Any, list[str], dict[str, Any]]:
    """Load either a plain sklearn model or a richer saved bundle."""
    loaded_object = joblib.load(model_path)
    if isinstance(loaded_object, dict):
        model = loaded_object.get("model")
        feature_columns = loaded_object.get("feature_columns", [])
        metadata = loaded_object.get("metadata", {})
        if model is None:
            raise ValueError("Saved model bundle is missing the model object.")
        return model, list(feature_columns), dict(metadata)

    model = loaded_object
    feature_columns = list(getattr(model, "feature_names_in_", []))
    return model, feature_columns, {}


def predict_event_strike_type(
    event_row,
    pose_df,
    model_path,
) -> dict:
    """Predict strike type for a single detected event using the trained baseline model."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Trained model not found: {model_path.as_posix()}")

    model, feature_columns, model_metadata = _load_model_bundle(model_path)
    feature_row, warnings = build_features_for_event_window(
        event_row=event_row,
        pose_df=pose_df,
        video_filename=str(getattr(event_row, "get", lambda *_: "")("video_filename", "")),
        clip_path=str(getattr(event_row, "get", lambda *_: "")("clip_path", "")),
        analysis_mode=str(getattr(event_row, "get", lambda *_: "single_athlete_training")("analysis_mode", "single_athlete_training")),
    )
    if feature_row is None:
        raise ValueError("; ".join(warnings) if warnings else "Unable to build prediction features.")

    feature_df = pd.DataFrame([feature_row])
    numeric_feature_df = feature_df.select_dtypes(include=["number", "bool"]).copy().fillna(0.0)

    if not feature_columns:
        feature_columns = list(getattr(model, "feature_names_in_", []))
    if not feature_columns:
        feature_columns = numeric_feature_df.columns.tolist()

    missing_columns = [
        column_name for column_name in feature_columns if column_name not in numeric_feature_df.columns
    ]
    extra_columns = [
        column_name for column_name in numeric_feature_df.columns if column_name not in feature_columns
    ]
    if missing_columns:
        warnings.append(
            f"{len(missing_columns)} model feature columns were missing from incoming features and filled with 0."
        )
    if extra_columns:
        warnings.append(
            f"{len(extra_columns)} incoming feature columns were not used by the saved model and were dropped."
        )

    aligned_feature_df = numeric_feature_df.reindex(columns=feature_columns, fill_value=0.0)
    suggested_strike_type = str(model.predict(aligned_feature_df)[0])

    prediction_confidence = None
    top_3_predictions: list[dict[str, float]] = []
    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        probabilities = model.predict_proba(aligned_feature_df)[0]
        top_predictions_df = pd.DataFrame(
            {
                "strike_type": model.classes_,
                "probability": probabilities,
            }
        ).sort_values("probability", ascending=False)
        top_predictions_df = top_predictions_df.head(3).reset_index(drop=True)
        top_3_predictions = [
            {
                "strike_type": str(row["strike_type"]),
                "probability": round(float(row["probability"]), 6),
            }
            for _, row in top_predictions_df.iterrows()
        ]
        if top_3_predictions:
            prediction_confidence = top_3_predictions[0]["probability"]

    return {
        "suggested_strike_type": suggested_strike_type,
        "prediction_confidence": prediction_confidence,
        "top_3_predictions": top_3_predictions,
        "feature_columns_used": feature_columns,
        "missing_feature_columns": missing_columns,
        "extra_feature_columns": extra_columns,
        "model_metadata": model_metadata,
        "warnings": warnings,
        "top_3_predictions_json": json.dumps(top_3_predictions),
    }
