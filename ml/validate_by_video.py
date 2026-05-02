from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ml.train_baseline_model import (
    DEFAULT_TARGET_COLUMN,
    load_selected_feature_columns,
)
from src.utils import (
    build_phase19_feature_dataset_csv_path,
    build_selected_fine_features_csv_path,
    build_video_holdout_confusion_csv_path,
    build_video_holdout_per_class_csv_path,
    build_video_holdout_summary_csv_path,
)


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


def _select_training_rows(feature_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Keep rows with usable target labels and source video names."""
    working_df = feature_df.copy()
    working_df[target_column] = working_df[target_column].astype(str).str.strip()
    working_df["video_filename"] = working_df["video_filename"].astype(str).str.strip()
    valid_mask = (
        working_df[target_column].notna()
        & (working_df[target_column] != "")
        & (working_df[target_column].str.lower() != "unknown")
        & (working_df[target_column].str.lower() != "nan")
        & working_df["video_filename"].notna()
        & (working_df["video_filename"] != "")
        & (working_df["video_filename"].str.lower() != "nan")
    )
    return working_df.loc[valid_mask].reset_index(drop=True)


def _align_features(
    feature_df: pd.DataFrame,
    selected_feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Align validation rows to the locked selected feature list."""
    numeric_df = feature_df.select_dtypes(include=["number", "bool"]).copy()
    feature_columns = [
        column_name
        for column_name in selected_feature_columns
        if column_name not in EXCLUDED_COLUMNS
    ]
    aligned_df = numeric_df.reindex(columns=feature_columns, fill_value=0.0).fillna(0.0)
    return aligned_df, feature_columns


def validate_by_heldout_video(
    dataset_path: str | Path = build_phase19_feature_dataset_csv_path(),
    selected_features_path: str | Path = build_selected_fine_features_csv_path(),
    summary_output_path: str | Path = build_video_holdout_summary_csv_path(),
    per_class_output_path: str | Path = build_video_holdout_per_class_csv_path(),
    confusion_output_path: str | Path = build_video_holdout_confusion_csv_path(),
    target_column: str = DEFAULT_TARGET_COLUMN,
    random_state: int = 42,
    min_test_rows: int = 1,
) -> dict[str, Any]:
    """Run leave-one-video-out validation using temporary fine strike models."""
    dataset_path = Path(dataset_path)
    selected_features_path = Path(selected_features_path)
    summary_output_path = Path(summary_output_path)
    per_class_output_path = Path(per_class_output_path)
    confusion_output_path = Path(confusion_output_path)

    if not dataset_path.exists():
        raise ValueError(f"Phase 19 feature dataset not found: {dataset_path.as_posix()}")

    feature_df = pd.read_csv(dataset_path)
    if feature_df.empty:
        raise ValueError("Phase 19 feature dataset exists but has no rows.")
    if "video_filename" not in feature_df.columns:
        raise ValueError("Phase 19 feature dataset is missing `video_filename`.")
    if target_column not in feature_df.columns:
        raise ValueError(f"Phase 19 feature dataset is missing `{target_column}`.")

    filtered_df = _select_training_rows(feature_df, target_column=target_column)
    if filtered_df.empty:
        raise ValueError("No labeled feature rows are available for video validation.")

    selected_feature_columns = load_selected_feature_columns(selected_features_path)
    labels = sorted(filtered_df[target_column].astype(str).unique().tolist())
    summary_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    warnings: list[str] = []

    video_counts = filtered_df["video_filename"].value_counts().sort_index()
    for heldout_video, test_row_count in video_counts.items():
        if int(test_row_count) < min_test_rows:
            warnings.append(
                f"Skipped {heldout_video}: only {int(test_row_count)} test rows."
            )
            continue

        train_df = filtered_df[filtered_df["video_filename"] != heldout_video].copy()
        test_df = filtered_df[filtered_df["video_filename"] == heldout_video].copy()
        if train_df.empty or test_df.empty:
            warnings.append(f"Skipped {heldout_video}: train or test split was empty.")
            continue
        if train_df[target_column].nunique() < 2:
            warnings.append(
                f"Skipped {heldout_video}: training split had fewer than 2 classes."
            )
            continue

        X_train, feature_columns = _align_features(train_df, selected_feature_columns)
        X_test, _ = _align_features(test_df, selected_feature_columns)
        y_train = train_df[target_column].astype(str)
        y_test = test_df[target_column].astype(str)

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
            min_samples_leaf=1,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        y_pred = pd.Series(model.predict(X_test), index=y_test.index)

        heldout_labels = sorted(y_test.unique().tolist())
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
        weighted_f1 = f1_score(
            y_test,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

        summary_rows.append(
            {
                "heldout_video": heldout_video,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_classes": int(y_train.nunique()),
                "test_classes": int(y_test.nunique()),
                "feature_count": int(len(feature_columns)),
                "accuracy": round(float(accuracy), 6),
                "macro_f1": round(float(macro_f1), 6),
                "weighted_f1": round(float(weighted_f1), 6),
            }
        )

        for label in heldout_labels:
            label_mask = y_test == label
            label_support = int(label_mask.sum())
            label_correct = int((y_pred[label_mask] == label).sum())
            per_class_rows.append(
                {
                    "heldout_video": heldout_video,
                    "label": label,
                    "support": label_support,
                    "correct": label_correct,
                    "recall": round(label_correct / label_support, 6)
                    if label_support
                    else 0.0,
                }
            )

        matrix = confusion_matrix(y_test, y_pred, labels=labels)
        for actual_index, actual_label in enumerate(labels):
            for predicted_index, predicted_label in enumerate(labels):
                count = int(matrix[actual_index][predicted_index])
                if count:
                    confusion_rows.append(
                        {
                            "heldout_video": heldout_video,
                            "actual_label": actual_label,
                            "predicted_label": predicted_label,
                            "count": count,
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    per_class_output_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output_path, index=False)
    per_class_df.to_csv(per_class_output_path, index=False)
    confusion_df.to_csv(confusion_output_path, index=False)

    return {
        "summary_df": summary_df,
        "per_class_df": per_class_df,
        "confusion_df": confusion_df,
        "warnings": warnings,
        "dataset_path": str(dataset_path),
        "selected_features_path": str(selected_features_path),
        "summary_output_path": str(summary_output_path),
        "per_class_output_path": str(per_class_output_path),
        "confusion_output_path": str(confusion_output_path),
        "dataset_rows": int(len(filtered_df)),
        "video_count": int(filtered_df["video_filename"].nunique()),
        "feature_count": int(len(selected_feature_columns)),
    }


if __name__ == "__main__":
    results = validate_by_heldout_video()
    print(
        f"Validated {results['video_count']} videos from {results['dataset_rows']} rows. "
        f"Summary saved to {results['summary_output_path']}"
    )
