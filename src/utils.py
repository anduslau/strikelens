from pathlib import Path


BASE_DATA_DIR = Path("data")
UPLOADS_DIR = BASE_DATA_DIR / "uploads"
PROCESSED_DIR = BASE_DATA_DIR / "processed"
CLIPS_DIR = PROCESSED_DIR / "clips"
LABELS_DIR = BASE_DATA_DIR / "labels"
EXPORTS_DIR = BASE_DATA_DIR / "exports"


def ensure_directories() -> None:
    """Create the required project folders if they do not already exist."""
    for directory in (UPLOADS_DIR, PROCESSED_DIR, CLIPS_DIR, LABELS_DIR, EXPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_processed_pose_csv_path(video_path: Path) -> Path:
    """Create a predictable CSV output path for extracted pose data."""
    return PROCESSED_DIR / f"{video_path.stem}_pose_data.csv"


def build_processed_overlay_video_path(video_path: Path) -> Path:
    """Create a predictable output path for an annotated pose overlay video."""
    return PROCESSED_DIR / f"{video_path.stem}_pose_overlay.mp4"


def build_processed_overlay_diagnostics_csv_path(video_path: Path) -> Path:
    """Create a predictable CSV output path for overlay diagnostics."""
    return PROCESSED_DIR / f"{video_path.stem}_pose_overlay_diagnostics.csv"


def build_processed_strike_events_csv_path(video_path: Path) -> Path:
    """Create a predictable CSV output path for detected strike events."""
    return PROCESSED_DIR / f"{video_path.stem}_strike_events.csv"


def build_processed_clips_dir(video_path: Path) -> Path:
    """Create a predictable directory path for exported event clips."""
    return CLIPS_DIR / video_path.stem


def build_event_labels_csv_path() -> Path:
    """Create a predictable CSV path for human review labels."""
    return LABELS_DIR / "event_labels.csv"


def build_clean_labeled_events_csv_path() -> Path:
    """Create a predictable CSV path for the cleaned labeled dataset export."""
    return EXPORTS_DIR / "clean_labeled_events.csv"


def build_ml_feature_dataset_csv_path() -> Path:
    """Create a predictable CSV path for the ML-ready feature dataset export."""
    return EXPORTS_DIR / "ml_feature_dataset.csv"


def build_phase19_feature_dataset_csv_path() -> Path:
    """Create the Phase 19 fixed feature dataset path."""
    return EXPORTS_DIR / "ml_feature_dataset_v6.csv"


def build_selected_fine_features_csv_path() -> Path:
    """Create the Phase 19 selected fine-strike feature list path."""
    return EXPORTS_DIR / "selected_features_corrected_strike_type_top_80_by_importance.csv"


def build_model_evaluation_csv_path() -> Path:
    """Create a predictable CSV path for baseline model evaluation metrics."""
    return EXPORTS_DIR / "model_evaluation.csv"


def build_video_holdout_summary_csv_path() -> Path:
    """Create a predictable CSV path for held-out-video validation summary metrics."""
    return EXPORTS_DIR / "video_holdout_validation_summary.csv"


def build_video_holdout_per_class_csv_path() -> Path:
    """Create a predictable CSV path for held-out-video per-class metrics."""
    return EXPORTS_DIR / "video_holdout_validation_per_class.csv"


def build_video_holdout_confusion_csv_path() -> Path:
    """Create a predictable CSV path for held-out-video confusion counts."""
    return EXPORTS_DIR / "video_holdout_validation_confusion.csv"


def build_baseline_model_path() -> Path:
    """Create a predictable file path for the saved baseline ML model."""
    return Path("ml") / "models" / "strike_type_random_forest.joblib"


def build_baseline_model_metadata_path() -> Path:
    """Create a predictable metadata path for the saved fine strike model."""
    return Path("ml") / "models" / "strike_type_random_forest_metadata.json"
