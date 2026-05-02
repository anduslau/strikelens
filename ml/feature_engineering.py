from pathlib import Path

import pandas as pd

from src.taxonomy import get_coarse_strike_type


KEY_LIMBS = [
    "left_wrist",
    "right_wrist",
    "left_elbow",
    "right_elbow",
    "left_ankle",
    "right_ankle",
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_shoulder",
    "right_shoulder",
]


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    """Normalize mixed boolean/string values loaded from CSV."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )


def _prepare_pose_dataframe(pose_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare pose data for stable numeric feature extraction."""
    working_df = pose_df.copy().sort_values("timestamp_seconds").reset_index(drop=True)
    working_df["timestamp_seconds"] = pd.to_numeric(
        working_df["timestamp_seconds"], errors="coerce"
    )
    working_df["average_visibility_score"] = pd.to_numeric(
        working_df.get("average_visibility_score", 0.0), errors="coerce"
    ).fillna(0.0)
    working_df["pose_detected"] = _normalize_bool_series(working_df["pose_detected"])

    for limb_name in KEY_LIMBS:
        for axis in ("x", "y"):
            column_name = f"{limb_name}_{axis}"
            working_df[column_name] = pd.to_numeric(
                working_df.get(column_name), errors="coerce"
            )

    return working_df


def _compute_limb_features(window_df: pd.DataFrame, limb_name: str) -> dict[str, float]:
    """Compute numeric movement features for a single limb over an event window."""
    x_series = window_df[f"{limb_name}_x"]
    y_series = window_df[f"{limb_name}_y"]
    time_delta = window_df["timestamp_seconds"].diff()
    valid_transition_mask = (
        window_df["pose_detected"]
        & window_df["pose_detected"].shift(1, fill_value=False)
        & x_series.notna()
        & x_series.shift(1).notna()
        & y_series.notna()
        & y_series.shift(1).notna()
    )

    x_delta = x_series.diff()
    y_delta = y_series.diff()
    distance = (x_delta.pow(2) + y_delta.pow(2)).pow(0.5)
    speed = distance.divide(time_delta).where(valid_transition_mask, other=0.0).fillna(0.0)

    valid_x = x_series.dropna()
    valid_y = y_series.dropna()
    if valid_x.empty or valid_y.empty:
        return {
            f"{limb_name}_max_speed": 0.0,
            f"{limb_name}_mean_speed": 0.0,
            f"{limb_name}_total_distance": 0.0,
            f"{limb_name}_x_displacement": 0.0,
            f"{limb_name}_y_displacement": 0.0,
            f"{limb_name}_max_x": 0.0,
            f"{limb_name}_min_x": 0.0,
            f"{limb_name}_max_y": 0.0,
            f"{limb_name}_min_y": 0.0,
            f"{limb_name}_range_x": 0.0,
            f"{limb_name}_range_y": 0.0,
        }

    x_displacement = float(valid_x.iloc[-1] - valid_x.iloc[0])
    y_displacement = float(valid_y.iloc[-1] - valid_y.iloc[0])
    max_x = float(valid_x.max())
    min_x = float(valid_x.min())
    max_y = float(valid_y.max())
    min_y = float(valid_y.min())

    return {
        f"{limb_name}_max_speed": round(float(speed.max()), 6),
        f"{limb_name}_mean_speed": round(float(speed.mean()), 6),
        f"{limb_name}_total_distance": round(float(distance.where(valid_transition_mask, other=0.0).fillna(0.0).sum()), 6),
        f"{limb_name}_x_displacement": round(x_displacement, 6),
        f"{limb_name}_y_displacement": round(y_displacement, 6),
        f"{limb_name}_max_x": round(max_x, 6),
        f"{limb_name}_min_x": round(min_x, 6),
        f"{limb_name}_max_y": round(max_y, 6),
        f"{limb_name}_min_y": round(min_y, 6),
        f"{limb_name}_range_x": round(max_x - min_x, 6),
        f"{limb_name}_range_y": round(max_y - min_y, 6),
    }


def build_features_for_event_window(
    event_row: pd.Series | dict[str, object],
    pose_df: pd.DataFrame,
    *,
    video_filename: str = "",
    clip_path: str = "",
    analysis_mode: str = "single_athlete_training",
    corrected_category: str = "unknown",
    corrected_strike_type: str = "unknown",
    pre_seconds: float = 0.25,
    post_seconds: float = 0.25,
) -> tuple[dict[str, object] | None, list[str]]:
    """Build the same event-level feature row used for training from pose/event inputs."""
    warnings: list[str] = []
    normalized_event_row = (
        event_row if isinstance(event_row, pd.Series) else pd.Series(event_row)
    )
    prepared_pose_df = _prepare_pose_dataframe(pose_df)

    try:
        start_time = max(float(normalized_event_row["start_time"]) - pre_seconds, 0.0)
        end_time = float(normalized_event_row["end_time"]) + post_seconds
    except Exception as exc:
        warnings.append(f"Invalid event timing for feature extraction: {exc}")
        return None, warnings

    window_df = prepared_pose_df[
        prepared_pose_df["timestamp_seconds"].between(start_time, end_time, inclusive="both")
    ].copy()
    if window_df.empty:
        warnings.append("No pose frames found in the requested event window.")
        return None, warnings

    predicted_confidence_score = pd.to_numeric(
        normalized_event_row.get("confidence_score", normalized_event_row.get("predicted_confidence_score", 0.0)),
        errors="coerce",
    )
    if pd.isna(predicted_confidence_score):
        predicted_confidence_score = 0.0

    predicted_primary_limb = str(
        normalized_event_row.get("primary_limb", normalized_event_row.get("predicted_primary_limb", "unknown"))
    )
    predicted_category = str(
        normalized_event_row.get("event_category", normalized_event_row.get("predicted_category", "unknown"))
    )
    event_id_value = pd.to_numeric(normalized_event_row.get("event_id", 0), errors="coerce")
    if pd.isna(event_id_value):
        event_id_value = 0

    feature_row: dict[str, object] = {
        "video_filename": video_filename,
        "analysis_mode": str(analysis_mode),
        "event_id": int(event_id_value),
        "clip_path": str(clip_path),
        "duration_seconds": round(
            float(normalized_event_row["end_time"]) - float(normalized_event_row["start_time"]),
            6,
        ),
        "predicted_confidence_score": round(float(predicted_confidence_score), 6),
        "predicted_primary_limb": predicted_primary_limb,
        "predicted_category": predicted_category,
        "average_visibility_score_mean": round(
            float(window_df["average_visibility_score"].mean()),
            6,
        ),
        "pose_lost_frame_count": int((~window_df["pose_detected"]).sum()),
        "pose_lost_ratio": round(float((~window_df["pose_detected"]).mean()), 6),
        "corrected_category": str(corrected_category),
        "corrected_strike_type": str(corrected_strike_type),
        "coarse_strike_type": get_coarse_strike_type(str(corrected_strike_type)),
    }

    for limb_name in KEY_LIMBS:
        feature_row.update(_compute_limb_features(window_df, limb_name))

    return feature_row, warnings


def build_features_for_labeled_event(
    label_row: pd.Series,
    processed_dir: str | Path,
    pre_seconds: float = 0.25,
    post_seconds: float = 0.25,
) -> tuple[dict[str, object] | None, list[str]]:
    """Join a cleaned label with event timing and pose data, then compute ML features."""
    warnings: list[str] = []
    processed_path = Path(processed_dir)
    video_filename = str(label_row["video_filename"])
    video_stem = Path(video_filename).stem
    pose_csv_path = processed_path / f"{video_stem}_pose_data.csv"
    strike_events_csv_path = processed_path / f"{video_stem}_strike_events.csv"

    if not pose_csv_path.exists():
        warnings.append(f"Missing pose CSV for {video_filename}: {pose_csv_path.name}")
        return None, warnings
    if not strike_events_csv_path.exists():
        warnings.append(
            f"Missing strike events CSV for {video_filename}: {strike_events_csv_path.name}"
        )
        return None, warnings

    try:
        pose_df = pd.read_csv(pose_csv_path)
        strike_events_df = pd.read_csv(strike_events_csv_path)
    except Exception as exc:
        warnings.append(f"Unable to load supporting files for {video_filename}: {exc}")
        return None, warnings

    event_id = int(label_row["event_id"])
    event_match = strike_events_df.loc[strike_events_df["event_id"] == event_id]
    if event_match.empty:
        warnings.append(f"Missing event_id {event_id} in {strike_events_csv_path.name}")
        return None, warnings

    event_row = event_match.iloc[0]
    feature_row, feature_warnings = build_features_for_event_window(
        event_row=event_row,
        pose_df=pose_df,
        video_filename=video_filename,
        clip_path=str(label_row["clip_path"]),
        analysis_mode=str(label_row.get("analysis_mode", "single_athlete_training")),
        corrected_category=str(label_row["corrected_category"]),
        corrected_strike_type=str(label_row["corrected_strike_type"]),
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
    )
    warnings.extend(feature_warnings)
    return feature_row, warnings
