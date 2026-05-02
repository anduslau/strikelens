from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def build_classification_report_dataframe(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
) -> pd.DataFrame:
    """Convert sklearn's classification report into a display-friendly DataFrame."""
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "label"})
    numeric_columns = ["precision", "recall", "f1-score", "support"]
    for column in numeric_columns:
        if column in report_df.columns:
            report_df[column] = pd.to_numeric(report_df[column], errors="coerce")
    return report_df


def build_confusion_matrix_dataframe(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
) -> pd.DataFrame:
    """Build a labeled confusion matrix DataFrame."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    matrix_df.index.name = "actual_label"
    return matrix_df.reset_index()


def build_feature_importance_dataframe(
    feature_names: list[str],
    importances: list[float],
) -> pd.DataFrame:
    """Build a sorted feature importance table."""
    feature_importance_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": importances,
        }
    )
    feature_importance_df["importance"] = pd.to_numeric(
        feature_importance_df["importance"],
        errors="coerce",
    ).fillna(0.0)
    return feature_importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


def build_model_metrics_export(
    report_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Flatten the model report into CSV-exportable rows with shared metadata."""
    export_df = report_df.copy()
    for key, value in metadata.items():
        export_df[key] = value
    preferred_order = [
        "run_timestamp",
        "model_name",
        "target_column",
        "evaluation_split",
        "stratified_split_used",
        "dataset_rows",
        "training_rows",
        "test_rows",
        "feature_count",
        "label",
        "precision",
        "recall",
        "f1-score",
        "support",
    ]
    existing_columns = [column for column in preferred_order if column in export_df.columns]
    remaining_columns = [column for column in export_df.columns if column not in existing_columns]
    return export_df[existing_columns + remaining_columns]
