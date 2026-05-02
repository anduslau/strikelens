from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.evaluate_model import (
    build_classification_report_dataframe,
    build_confusion_matrix_dataframe,
    build_feature_importance_dataframe,
    build_model_metrics_export,
)
from src.utils import (
    build_baseline_model_path,
    build_baseline_model_metadata_path,
    build_model_evaluation_csv_path,
    build_phase19_feature_dataset_csv_path,
    build_selected_fine_features_csv_path,
)


DEFAULT_TARGET_COLUMN = "corrected_strike_type"
EXCLUDED_COLUMNS = {
    "video_filename",
    "clip_path",
    "corrected_category",
    "corrected_strike_type",
    "coarse_strike_type",
    "notes",
    "created_at",
    "event_id",
}
MODEL_NAME_BY_TARGET = {
    "corrected_strike_type": "strike_type_random_forest",
}
SELECTED_FEATURE_COUNT = 80


def _select_training_rows(feature_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Keep only rows that have a usable target label."""
    working_df = feature_df.copy()
    working_df[target_column] = working_df[target_column].astype(str).str.strip()
    valid_mask = (
        working_df[target_column].notna()
        & (working_df[target_column] != "")
        & (working_df[target_column].str.lower() != "unknown")
        & (working_df[target_column].str.lower() != "nan")
    )
    return working_df.loc[valid_mask].reset_index(drop=True)


def load_selected_feature_columns(
    selected_features_path: str | Path = build_selected_fine_features_csv_path(),
) -> list[str]:
    """Load the fixed Phase 19 fine-strike feature list."""
    selected_features_path = Path(selected_features_path)
    if not selected_features_path.exists():
        raise ValueError(
            f"Selected feature list not found: {selected_features_path.as_posix()}"
        )

    selected_features_df = pd.read_csv(selected_features_path)
    if selected_features_df.empty:
        raise ValueError("Selected feature list exists but has no rows.")

    preferred_columns = [
        "feature_name",
        "feature",
        "column",
        "column_name",
    ]
    feature_column = next(
        (
            column_name
            for column_name in preferred_columns
            if column_name in selected_features_df.columns
        ),
        selected_features_df.columns[0],
    )
    feature_columns = (
        selected_features_df[feature_column]
        .astype(str)
        .str.strip()
        .loc[lambda series: (series != "") & (series.str.lower() != "nan")]
        .head(SELECTED_FEATURE_COUNT)
        .tolist()
    )
    feature_columns = list(dict.fromkeys(feature_columns))

    if len(feature_columns) != SELECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {SELECTED_FEATURE_COUNT} selected features, found {len(feature_columns)}."
        )

    return feature_columns


def _select_numeric_feature_columns(
    feature_df: pd.DataFrame,
    selected_feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Align training data to the fixed selected numeric feature columns."""
    numeric_df = feature_df.select_dtypes(include=["number", "bool"]).copy()
    feature_columns = [
        column_name for column_name in selected_feature_columns if column_name not in EXCLUDED_COLUMNS
    ]
    missing_columns = [
        column_name for column_name in feature_columns if column_name not in numeric_df.columns
    ]
    training_features_df = (
        numeric_df.reindex(columns=feature_columns, fill_value=0.0).copy().fillna(0.0)
    )
    return training_features_df, feature_columns, missing_columns


def _can_use_stratified_split(target_series: pd.Series, test_size: float) -> bool:
    """Use stratification only if each class can appear in both train and test sets."""
    class_counts = target_series.value_counts()
    if len(class_counts) < 2:
        return False

    min_class_count = int(class_counts.min())
    estimated_test_rows = max(1, int(round(len(target_series) * test_size)))
    return min_class_count >= 2 and estimated_test_rows >= len(class_counts)


def train_baseline_model(
    dataset_path: str | Path = build_phase19_feature_dataset_csv_path(),
    target_column: str = DEFAULT_TARGET_COLUMN,
    model_output_path: str | Path | None = None,
    metrics_output_path: str | Path | None = None,
    metadata_output_path: str | Path | None = None,
    selected_features_path: str | Path = build_selected_fine_features_csv_path(),
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train and evaluate the Phase 19 fine strike RandomForest model."""
    dataset_path = Path(dataset_path)
    if target_column not in MODEL_NAME_BY_TARGET:
        raise ValueError(f"Unsupported training target: {target_column}")

    model_output = Path(model_output_path) if model_output_path else build_baseline_model_path()
    metrics_output = Path(metrics_output_path) if metrics_output_path else build_model_evaluation_csv_path()
    metadata_output = (
        Path(metadata_output_path)
        if metadata_output_path
        else build_baseline_model_metadata_path()
    )
    model_name = MODEL_NAME_BY_TARGET[target_column]

    if not dataset_path.exists():
        raise ValueError(f"ML feature dataset not found: {dataset_path.as_posix()}")

    feature_df = pd.read_csv(dataset_path)
    if feature_df.empty:
        raise ValueError("ML feature dataset exists but has no rows.")
    if target_column not in feature_df.columns:
        raise ValueError(
            f"ML feature dataset is missing the required target column: {target_column}"
        )

    filtered_df = _select_training_rows(feature_df, target_column=target_column)
    if filtered_df.empty:
        raise ValueError(
            f"No training rows remain after filtering out null or unknown values for {target_column}."
        )

    selected_feature_columns = load_selected_feature_columns(selected_features_path)
    X, feature_columns, missing_feature_columns = _select_numeric_feature_columns(
        filtered_df,
        selected_feature_columns=selected_feature_columns,
    )
    y = filtered_df[target_column].astype(str)
    if X.empty:
        raise ValueError("No numeric feature columns are available for model training.")

    class_distribution_df = y.value_counts().reset_index()
    class_distribution_df.columns = [target_column, "count"]
    warnings: list[str] = []

    if len(filtered_df) < 20:
        warnings.append(
            "The current dataset is very small. Results should be treated as directional only."
        )
    if missing_feature_columns:
        warnings.append(
            f"{len(missing_feature_columns)} selected feature columns were missing from the training dataset and filled with 0."
        )

    if y.nunique() < 2:
        warnings.append(
            "Only one strike-type class is present. The model can be trained, but evaluation is performed on the same rows."
        )

    evaluation_split = "holdout"
    stratified_split_used = False

    if len(filtered_df) < 4 or y.nunique() < 2:
        X_train = X.copy()
        X_test = X.copy()
        y_train = y.copy()
        y_test = y.copy()
        evaluation_split = "training_only"
        warnings.append(
            "Evaluation is using the training rows because the dataset is too small for a meaningful holdout split."
        )
    else:
        stratify = None
        if _can_use_stratified_split(y, test_size=test_size):
            stratify = y
            stratified_split_used = True
        else:
            warnings.append(
                "A simple train/test split was used because there are not enough examples per class for a stable stratified split."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        min_samples_leaf=1,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    y_pred = pd.Series(model.predict(X_test), index=y_test.index, name="predicted_label")
    labels = sorted(y.unique().tolist())
    classification_report_df = build_classification_report_dataframe(y_test, y_pred, labels)
    confusion_matrix_df = build_confusion_matrix_dataframe(y_test, y_pred, labels)
    feature_importance_df = build_feature_importance_dataframe(
        feature_names=feature_columns,
        importances=model.feature_importances_.tolist(),
    )

    run_timestamp = datetime.now().isoformat(timespec="seconds")
    metrics_export_df = build_model_metrics_export(
        report_df=classification_report_df,
        metadata={
            "run_timestamp": run_timestamp,
            "model_name": model_name,
            "target_column": target_column,
            "evaluation_split": evaluation_split,
            "stratified_split_used": stratified_split_used,
            "dataset_rows": int(len(filtered_df)),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_count": int(len(feature_columns)),
        },
    )

    model_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_timestamp": run_timestamp,
        "model_name": model_name,
        "target_column": target_column,
        "dataset_path": dataset_path.as_posix(),
        "selected_features_path": Path(selected_features_path).as_posix(),
        "feature_columns": feature_columns,
        "feature_count": int(len(feature_columns)),
        "missing_training_feature_columns": missing_feature_columns,
        "dataset_rows": int(len(filtered_df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "evaluation_split": evaluation_split,
        "stratified_split_used": stratified_split_used,
    }
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "metadata": metadata,
            "trained_at": run_timestamp,
            "target_column": target_column,
        },
        model_output,
    )
    metrics_export_df.to_csv(metrics_output, index=False)
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "dataset_rows": int(len(filtered_df)),
        "feature_count": int(len(feature_columns)),
        "class_distribution_df": class_distribution_df,
        "warnings": warnings,
        "model_name": model_name,
        "model_path": str(model_output),
        "metrics_path": str(metrics_output),
        "metadata_path": str(metadata_output),
        "evaluation_split": evaluation_split,
        "stratified_split_used": stratified_split_used,
        "classification_report_df": classification_report_df,
        "confusion_matrix_df": confusion_matrix_df,
        "feature_importance_df": feature_importance_df,
        "top_feature_importances_df": feature_importance_df.head(20).copy(),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_column": target_column,
    }


if __name__ == "__main__":
    results = train_baseline_model()
    print(
        f"Trained {results['model_name']} on {results['dataset_rows']} rows with "
        f"{results['feature_count']} numeric features."
    )
    print(f"Model saved to {results['model_path']}")
    print(f"Metrics saved to {results['metrics_path']}")
