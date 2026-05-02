import math

import pandas as pd


PRIMARY_LIMBS = {
    "left_wrist": "punch_candidate",
    "right_wrist": "punch_candidate",
    "left_ankle": "kick_candidate",
    "right_ankle": "kick_candidate",
}

SUPPORT_LIMBS = ["left_elbow", "right_elbow", "left_knee", "right_knee"]
DIAGNOSTIC_LIMBS = [
    "left_wrist",
    "right_wrist",
    "left_ankle",
    "right_ankle",
    "left_knee",
    "right_knee",
]
DEFAULT_SPEED_THRESHOLD = 0.9
DEFAULT_MIN_EVENT_DURATION_SECONDS = 0.12
DEFAULT_MERGE_GAP_SECONDS = 0.12
DEFAULT_MIN_VISIBILITY_THRESHOLD = 0.6
DEFAULT_SMOOTHING_WINDOW = 3
MIN_DISPLACEMENT = 0.03


def _validate_pose_dataframe(pose_df: pd.DataFrame) -> None:
    """Ensure the pose dataframe contains the columns needed for generic strike detection."""
    required_columns = {
        "frame_number",
        "timestamp_seconds",
        "pose_detected",
        "average_visibility_score",
    }

    for limb_name in list(PRIMARY_LIMBS) + SUPPORT_LIMBS:
        required_columns.update({f"{limb_name}_x", f"{limb_name}_y"})

    missing_columns = sorted(required_columns - set(pose_df.columns))
    if missing_columns:
        raise ValueError(
            "Pose data is missing required columns: " + ", ".join(missing_columns)
        )


def _prepare_pose_dataframe(pose_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types so movement calculations are stable and deterministic."""
    working_df = pose_df.copy().sort_values("frame_number").reset_index(drop=True)

    numeric_columns = ["frame_number", "timestamp_seconds", "average_visibility_score"]
    for limb_name in list(PRIMARY_LIMBS) + SUPPORT_LIMBS:
        numeric_columns.extend([f"{limb_name}_x", f"{limb_name}_y"])

    for column_name in numeric_columns:
        working_df[column_name] = pd.to_numeric(working_df[column_name], errors="coerce")

    working_df["pose_detected"] = (
        working_df["pose_detected"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )
    working_df["average_visibility_score"] = working_df["average_visibility_score"].fillna(0.0)
    return working_df


def _estimate_frame_interval_seconds(working_df: pd.DataFrame) -> float:
    """Estimate frame spacing from timestamps so duration thresholds can stay in seconds."""
    frame_intervals = working_df["timestamp_seconds"].diff()
    positive_intervals = frame_intervals[frame_intervals > 0]

    if positive_intervals.empty:
        raise ValueError("Unable to estimate frame timing from pose timestamps.")

    return float(positive_intervals.median())


def _seconds_to_frames(duration_seconds: float, frame_interval_seconds: float) -> int:
    """Convert a duration in seconds into a frame count."""
    return max(1, int(math.ceil(duration_seconds / frame_interval_seconds)))


def _add_limb_speed_columns(working_df: pd.DataFrame, smoothing_window: int) -> pd.DataFrame:
    """Compute raw and smoothed x/y movement speeds in normalized coordinates per second."""
    time_delta = working_df["timestamp_seconds"].diff().replace(0, float("nan"))
    valid_transition_mask = working_df["pose_detected"] & working_df["pose_detected"].shift(
        1, fill_value=False
    )

    for limb_name in list(PRIMARY_LIMBS) + SUPPORT_LIMBS:
        x_delta = working_df[f"{limb_name}_x"].diff()
        y_delta = working_df[f"{limb_name}_y"].diff()
        # MediaPipe x/y coordinates are normalized to the frame, so these speeds
        # are measured in normalized coordinate units per second.
        distance = (x_delta.pow(2) + y_delta.pow(2)).pow(0.5)
        speed = distance.divide(time_delta)
        speed = speed.where(valid_transition_mask, other=0.0).fillna(0.0)
        working_df[f"{limb_name}_speed"] = speed
        working_df[f"{limb_name}_speed_smooth"] = (
            speed.rolling(window=smoothing_window, min_periods=1, center=True).mean()
        )

    return working_df


def _merge_event_windows(
    event_windows: list[dict[str, int | str]],
    merge_gap_frames: int,
) -> list[dict[str, int | str]]:
    """Merge nearby windows from the same limb so one strike is not counted twice."""
    if not event_windows:
        return []

    event_windows = sorted(event_windows, key=lambda item: (item["primary_limb"], item["start_idx"]))
    merged_windows: list[dict[str, int | str]] = [event_windows[0].copy()]

    for current_window in event_windows[1:]:
        previous_window = merged_windows[-1]
        same_limb = current_window["primary_limb"] == previous_window["primary_limb"]
        close_enough = current_window["start_idx"] - previous_window["end_idx"] <= merge_gap_frames

        if same_limb and close_enough:
            previous_window["end_idx"] = max(previous_window["end_idx"], current_window["end_idx"])
        else:
            merged_windows.append(current_window.copy())

    return merged_windows


def _build_event_note(
    has_pose_loss: bool,
    visibility_mean: float,
    support_peak: float,
    min_visibility_threshold: float,
) -> str:
    """Create a compact note explaining why confidence may be reduced."""
    notes: list[str] = []

    if has_pose_loss:
        notes.append("overlaps pose-lost frames")
    if visibility_mean < min_visibility_threshold:
        notes.append("low visibility reduced confidence")
    if support_peak < 0.35:
        notes.append("limited supporting joint movement")

    return "; ".join(notes) if notes else "generic strike-like movement candidate"


def prepare_strike_detection_analysis(
    pose_df: pd.DataFrame,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    min_event_duration_seconds: float = DEFAULT_MIN_EVENT_DURATION_SECONDS,
    merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS,
    min_visibility_threshold: float = DEFAULT_MIN_VISIBILITY_THRESHOLD,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> dict[str, object]:
    """Prepare per-frame speed features and diagnostics for strike detection tuning."""
    if speed_threshold <= 0:
        raise ValueError("Speed threshold must be greater than 0.")
    if min_event_duration_seconds <= 0:
        raise ValueError("Minimum event duration must be greater than 0.")
    if merge_gap_seconds < 0:
        raise ValueError("Merge gap cannot be negative.")
    if not 0 <= min_visibility_threshold <= 1:
        raise ValueError("Minimum visibility threshold must be between 0 and 1.")
    if smoothing_window < 1:
        raise ValueError("Smoothing window must be at least 1.")

    _validate_pose_dataframe(pose_df)
    working_df = _prepare_pose_dataframe(pose_df)
    frame_interval_seconds = _estimate_frame_interval_seconds(working_df)
    min_event_frames = _seconds_to_frames(min_event_duration_seconds, frame_interval_seconds)
    merge_gap_frames = _seconds_to_frames(merge_gap_seconds, frame_interval_seconds)
    working_df = _add_limb_speed_columns(working_df, smoothing_window=smoothing_window)

    diagnostics_summary: dict[str, float | int] = {}
    max_observed_limb_speed = 0.0

    for limb_name in DIAGNOSTIC_LIMBS:
        max_speed = float(working_df[f"{limb_name}_speed_smooth"].max())
        max_observed_limb_speed = max(max_observed_limb_speed, max_speed)
        frames_above_threshold = int((working_df[f"{limb_name}_speed_smooth"] >= speed_threshold).sum())
        diagnostics_summary[f"max_{limb_name}_speed"] = round(max_speed, 4)
        diagnostics_summary[f"{limb_name}_frames_above_threshold"] = frames_above_threshold

    return {
        "working_df": working_df,
        "diagnostics_df": pd.DataFrame([diagnostics_summary]),
        "frame_interval_seconds": frame_interval_seconds,
        "min_event_frames": min_event_frames,
        "merge_gap_frames": merge_gap_frames,
        "max_observed_limb_speed": round(max_observed_limb_speed, 4),
        "speed_threshold": speed_threshold,
        "min_visibility_threshold": min_visibility_threshold,
    }


def detect_strike_events(
    pose_df: pd.DataFrame,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    min_event_duration_seconds: float = DEFAULT_MIN_EVENT_DURATION_SECONDS,
    merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS,
    min_visibility_threshold: float = DEFAULT_MIN_VISIBILITY_THRESHOLD,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> pd.DataFrame:
    """Detect generic punch/kick candidates from smoothed limb movement only."""
    analysis = prepare_strike_detection_analysis(
        pose_df=pose_df,
        speed_threshold=speed_threshold,
        min_event_duration_seconds=min_event_duration_seconds,
        merge_gap_seconds=merge_gap_seconds,
        min_visibility_threshold=min_visibility_threshold,
        smoothing_window=smoothing_window,
    )
    working_df = analysis["working_df"]
    min_event_frames = int(analysis["min_event_frames"])
    merge_gap_frames = int(analysis["merge_gap_frames"])

    event_windows: list[dict[str, int | str]] = []

    for primary_limb in PRIMARY_LIMBS:
        active_mask = (working_df[f"{primary_limb}_speed_smooth"] >= speed_threshold).tolist()

        start_idx: int | None = None
        for row_index, is_active in enumerate(active_mask):
            if is_active and start_idx is None:
                start_idx = row_index
                continue

            if not is_active and start_idx is not None:
                end_idx = row_index - 1
                if end_idx - start_idx + 1 >= min_event_frames:
                    event_windows.append(
                        {
                            "primary_limb": primary_limb,
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                        }
                    )
                start_idx = None

        if start_idx is not None:
            end_idx = len(active_mask) - 1
            if end_idx - start_idx + 1 >= min_event_frames:
                event_windows.append(
                    {
                        "primary_limb": primary_limb,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                    }
                )

    detected_events: list[dict[str, int | float | str]] = []
    merged_windows = _merge_event_windows(event_windows, merge_gap_frames=merge_gap_frames)

    for event_id, event_window in enumerate(merged_windows, start=1):
        primary_limb = str(event_window["primary_limb"])
        event_slice = working_df.iloc[
            int(event_window["start_idx"]) : int(event_window["end_idx"]) + 1
        ].copy()
        primary_speed_series = event_slice[f"{primary_limb}_speed_smooth"]
        peak_speed = float(primary_speed_series.max())

        if math.isnan(peak_speed) or peak_speed < speed_threshold:
            continue

        displacement = math.sqrt(
            float(event_slice[f"{primary_limb}_x"].max() - event_slice[f"{primary_limb}_x"].min()) ** 2
            + float(event_slice[f"{primary_limb}_y"].max() - event_slice[f"{primary_limb}_y"].min()) ** 2
        )
        if displacement < MIN_DISPLACEMENT:
            continue

        peak_index = int(primary_speed_series.idxmax())
        peak_row = working_df.loc[peak_index]
        start_row = event_slice.iloc[0]
        end_row = event_slice.iloc[-1]

        related_support_limb = (
            "left_elbow"
            if primary_limb == "left_wrist"
            else "right_elbow"
            if primary_limb == "right_wrist"
            else "left_knee"
            if primary_limb == "left_ankle"
            else "right_knee"
        )
        support_peak = float(event_slice[f"{related_support_limb}_speed_smooth"].max())
        context_start = max(int(event_window["start_idx"]) - 1, 0)
        context_end = min(int(event_window["end_idx"]) + 1, len(working_df) - 1)
        context_slice = working_df.iloc[context_start : context_end + 1]
        pose_loss_overlap = bool((~context_slice["pose_detected"]).any())
        visibility_mean = float(event_slice["average_visibility_score"].mean())

        normalized_peak = min(peak_speed / max(speed_threshold * 1.5, 1e-6), 1.0)
        normalized_displacement = min(displacement / 0.2, 1.0)
        visibility_factor = min(max(visibility_mean, 0.0), 1.0)
        support_factor = min(max(support_peak / max(speed_threshold, 1e-6), 0.0), 1.0)
        confidence_score = (
            0.45 * normalized_peak
            + 0.2 * normalized_displacement
            + 0.2 * visibility_factor
            + 0.15 * support_factor
        )

        if pose_loss_overlap:
            confidence_score *= 0.75
        if visibility_mean < min_visibility_threshold:
            confidence_score *= 0.85

        detected_events.append(
            {
                "event_id": event_id,
                "start_frame": int(start_row["frame_number"]),
                "end_frame": int(end_row["frame_number"]),
                "start_time": round(float(start_row["timestamp_seconds"]), 4),
                "end_time": round(float(end_row["timestamp_seconds"]), 4),
                "peak_frame": int(peak_row["frame_number"]),
                "peak_time": round(float(peak_row["timestamp_seconds"]), 4),
                "primary_limb": primary_limb,
                "event_category": PRIMARY_LIMBS[primary_limb],
                "confidence_score": round(min(confidence_score, 1.0), 4),
                "notes": _build_event_note(
                    has_pose_loss=pose_loss_overlap,
                    visibility_mean=visibility_mean,
                    support_peak=support_peak,
                    min_visibility_threshold=min_visibility_threshold,
                ),
            }
        )

    return pd.DataFrame(
        detected_events,
        columns=[
            "event_id",
            "start_frame",
            "end_frame",
            "start_time",
            "end_time",
            "peak_frame",
            "peak_time",
            "primary_limb",
            "event_category",
            "confidence_score",
            "notes",
        ],
    )
