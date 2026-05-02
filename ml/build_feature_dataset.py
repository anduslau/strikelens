from pathlib import Path

import pandas as pd

from ml.feature_engineering import build_features_for_labeled_event


def build_feature_dataset(
    clean_labels_path: str | Path = "data/exports/clean_labeled_events.csv",
    processed_dir: str | Path = "data/processed",
    output_path: str | Path = "data/exports/ml_feature_dataset.csv",
) -> tuple[pd.DataFrame, list[str]]:
    """Build the ML-ready feature dataset from cleaned labels and saved pose/event data."""
    clean_labels_path = Path(clean_labels_path)
    processed_dir = Path(processed_dir)
    output_path = Path(output_path)

    if not clean_labels_path.exists():
        raise ValueError(f"Clean labels file not found: {clean_labels_path.as_posix()}")

    clean_labels_df = pd.read_csv(clean_labels_path)
    if clean_labels_df.empty:
        raise ValueError("Clean labels file exists but has no rows to process.")

    feature_rows: list[dict[str, object]] = []
    warnings: list[str] = []

    for _, label_row in clean_labels_df.iterrows():
        feature_row, row_warnings = build_features_for_labeled_event(
            label_row=label_row,
            processed_dir=processed_dir,
            pre_seconds=0.25,
            post_seconds=0.25,
        )
        warnings.extend(row_warnings)
        if feature_row is not None:
            feature_rows.append(feature_row)

    feature_dataset_df = pd.DataFrame(feature_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_dataset_df.to_csv(output_path, index=False)
    return feature_dataset_df, warnings


if __name__ == "__main__":
    dataset_df, dataset_warnings = build_feature_dataset()
    print(f"Built feature dataset with {len(dataset_df)} rows.")
    if dataset_warnings:
        print("Warnings:")
        for warning in dataset_warnings:
            print(f"- {warning}")
