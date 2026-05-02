import json
import altair as alt
import pandas as pd
import streamlit as st
import shutil
from hashlib import md5
from datetime import datetime
from pathlib import Path

from ml.build_feature_dataset import build_feature_dataset
from ml.predict_strike_type import predict_event_strike_type
from ml.train_baseline_model import train_baseline_model
from ml.validate_by_video import validate_by_heldout_video
from src.clip_exporter import export_event_clips
from src.pose_extractor import extract_pose_from_video_with_progress
from src.strike_detector import (
    DEFAULT_MERGE_GAP_SECONDS,
    DEFAULT_MIN_EVENT_DURATION_SECONDS,
    DEFAULT_MIN_VISIBILITY_THRESHOLD,
    DEFAULT_SMOOTHING_WINDOW,
    DEFAULT_SPEED_THRESHOLD,
    detect_strike_events,
)
from src.taxonomy import get_coarse_strike_type
from src.utils import (
    UPLOADS_DIR,
    build_baseline_model_metadata_path,
    build_baseline_model_path,
    build_clean_labeled_events_csv_path,
    build_event_labels_csv_path,
    build_model_evaluation_csv_path,
    build_phase19_feature_dataset_csv_path,
    build_selected_fine_features_csv_path,
    build_processed_clips_dir,
    build_processed_overlay_video_path,
    build_processed_pose_csv_path,
    build_processed_strike_events_csv_path,
    build_video_holdout_confusion_csv_path,
    build_video_holdout_per_class_csv_path,
    build_video_holdout_summary_csv_path,
    ensure_directories,
)
from src.video_loader import (
    extract_video_metadata,
    format_metadata,
    is_valid_video_file,
    save_uploaded_video,
)
from src.visualization import create_pose_overlay_video_with_diagnostics


CORRECTED_CATEGORIES = [
    "punch",
    "kick",
    "defense",
    "footwork",
    "false_positive",
    "unknown",
]
PUNCH_STRIKE_TYPES = [
    "jab",
    "cross",
    "hook",
    "uppercut",
]
KICK_STRIKE_TYPES = [
    "roundhouse_kick",
    "axe_kick",
    "back_kick",
    "cut_kick",
    "double_kick",
    "frontdouble_kick",
    "hopstep_kick",
    "spinninghook_kick",
    "tornado_kick",
    "hopaxe_kick",
    "cheapshot_kick",
    "crescentaxe_kick",
]
CORRECTED_STRIKE_TYPES = [
    *PUNCH_STRIKE_TYPES,
    *KICK_STRIKE_TYPES,
    "unknown",
]


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    """Normalize mixed boolean/string values from CSV labels into True/False."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )


def _format_strike_type_option(strike_type: str) -> str:
    """Show grouped strike labels while keeping raw saved values unchanged."""
    if strike_type in PUNCH_STRIKE_TYPES:
        return f"Punch: {strike_type}"
    if strike_type in KICK_STRIKE_TYPES:
        return f"Kick: {strike_type}"
    return strike_type


def save_event_labels(
    video_filename: str,
    analysis_mode: str,
    clips_df: pd.DataFrame,
    labels_csv_path: str,
) -> tuple[int, int]:
    """Append human review labels for exported clips to the labels CSV."""
    if clips_df.empty:
        raise ValueError("No exported clips are available to label.")

    created_at = datetime.now().isoformat(timespec="seconds")
    label_rows: list[dict[str, str | int | float | bool]] = []

    for clip in clips_df.to_dict(orient="records"):
        event_id = int(clip["event_id"])
        clip_path = str(clip["clip_path"])
        corrected_category_key = _build_review_state_key(
            "corrected_category",
            event_id,
            clip_path,
        )
        corrected_strike_type_key = _build_review_state_key(
            "corrected_strike_type",
            event_id,
            clip_path,
        )
        valid_strike_key = _build_review_state_key(
            "valid_strike",
            event_id,
            clip_path,
        )
        notes_key = _build_review_state_key(
            "notes",
            event_id,
            clip_path,
        )
        label_rows.append(
            {
                "video_filename": video_filename,
                "analysis_mode": str(
                    clip.get("analysis_mode", analysis_mode)
                ),
                "event_id": event_id,
                "clip_path": str(clip["clip_path"]),
                "predicted_category": str(clip["event_category"]),
                "predicted_primary_limb": str(clip["primary_limb"]),
                "predicted_confidence_score": float(clip["confidence_score"]),
                "model_suggested_strike_type": str(
                    clip.get("model_suggested_strike_type", "")
                ),
                "model_prediction_confidence": clip.get(
                    "model_prediction_confidence", None
                ),
                "model_top_3_predictions": str(
                    clip.get("model_top_3_predictions", "")
                ),
                "corrected_category": st.session_state.get(
                    corrected_category_key,
                    "unknown",
                ),
                "corrected_strike_type": st.session_state.get(
                    corrected_strike_type_key,
                    "unknown",
                ),
                "is_valid_strike": bool(st.session_state.get(valid_strike_key, False)),
                "notes": st.session_state.get(notes_key, ""),
                "created_at": created_at,
            }
        )

    labels_df = pd.DataFrame(label_rows)
    labels_path = Path(labels_csv_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    dedupe_keys = ["video_filename", "event_id", "clip_path"]
    attempted_rows = len(label_rows)

    if labels_path.exists():
        existing_df = pd.read_csv(labels_path)
        existing_count = len(existing_df)
        combined_df = pd.concat([existing_df, labels_df], ignore_index=True)
        labels_df = combined_df.drop_duplicates(subset=dedupe_keys, keep="last")
    else:
        existing_count = 0
        labels_df = labels_df.drop_duplicates(subset=dedupe_keys, keep="last")

    labels_df.to_csv(labels_path, index=False)
    added_rows = max(len(labels_df) - existing_count, 0)
    return added_rows, attempted_rows


def _build_review_state_key(base_key: str, event_id: int, clip_path: str) -> str:
    """Create a stable session-state key for a specific review clip."""
    clip_hash = md5(clip_path.encode("utf-8")).hexdigest()[:10]
    return f"{base_key}_{event_id}_{clip_hash}"


def _initialize_clip_review_state(clip: dict[str, object]) -> dict[str, str]:
    """Seed review widgets once without overwriting later user edits."""
    event_id = int(clip["event_id"])
    clip_path = str(clip["clip_path"])
    suggested_strike_type = str(clip.get("model_suggested_strike_type", "")).strip()

    corrected_category_key = _build_review_state_key(
        "corrected_category",
        event_id,
        clip_path,
    )
    corrected_strike_type_key = _build_review_state_key(
        "corrected_strike_type",
        event_id,
        clip_path,
    )
    valid_strike_key = _build_review_state_key(
        "valid_strike",
        event_id,
        clip_path,
    )
    notes_key = _build_review_state_key(
        "notes",
        event_id,
        clip_path,
    )

    if corrected_category_key not in st.session_state:
        st.session_state[corrected_category_key] = "unknown"
    if st.session_state[corrected_category_key] not in CORRECTED_CATEGORIES:
        st.session_state[corrected_category_key] = "unknown"

    if corrected_strike_type_key not in st.session_state:
        st.session_state[corrected_strike_type_key] = (
            suggested_strike_type
            if suggested_strike_type in CORRECTED_STRIKE_TYPES
            else "unknown"
        )
    if st.session_state[corrected_strike_type_key] not in CORRECTED_STRIKE_TYPES:
        st.session_state[corrected_strike_type_key] = "unknown"

    if valid_strike_key not in st.session_state:
        st.session_state[valid_strike_key] = False
    if notes_key not in st.session_state:
        st.session_state[notes_key] = ""

    return {
        "corrected_category_key": corrected_category_key,
        "corrected_strike_type_key": corrected_strike_type_key,
        "valid_strike_key": valid_strike_key,
        "notes_key": notes_key,
    }


def render_clip_review_section(clips_df: pd.DataFrame) -> None:
    """Render one review block per exported event clip."""
    st.subheader("Review Event Clips")

    for clip in clips_df.to_dict(orient="records"):
        event_id = int(clip["event_id"])
        clip_path = Path(str(clip["clip_path"]))
        state_keys = _initialize_clip_review_state(clip)
        suggested_strike_type = str(clip.get("model_suggested_strike_type", "")).strip()
        prediction_confidence = pd.to_numeric(
            clip.get("model_prediction_confidence"),
            errors="coerce",
        )
        top_3_predictions = str(clip.get("model_top_3_predictions", "")).strip()

        with st.container(border=True):
            st.markdown(f"**Event {event_id}**")
            review_col, meta_col = st.columns([1.5, 1])

            with review_col:
                if not clip_path.exists():
                    st.warning("Clip file does not exist.")
                else:
                    file_size_bytes = clip_path.stat().st_size
                    if file_size_bytes <= 0:
                        st.warning("Clip file size is 0 bytes.")
                    else:
                        with clip_path.open("rb") as video_file:
                            video_bytes = video_file.read()
                        st.video(video_bytes, format="video/mp4")

                    if clip.get("codec_source") == "opencv_raw":
                        st.warning(
                            "FFmpeg was unavailable or conversion failed. Browser playback may fail for this raw clip."
                        )

                if suggested_strike_type:
                    st.warning(
                        "Review the model suggestion before saving the label."
                    )
                    st.write(f"Suggested strike type: `{suggested_strike_type}`")
                    if pd.notna(prediction_confidence):
                        st.write(f"Model confidence: `{float(prediction_confidence):.4f}`")
                    if top_3_predictions:
                        st.caption(f"Top 3 predictions: `{top_3_predictions}`")

            with meta_col:
                st.write(f"Predicted category: `{clip['event_category']}`")
                st.write(f"Primary limb: `{clip['primary_limb']}`")
                st.write(f"Confidence score: `{float(clip['confidence_score']):.4f}`")
                st.checkbox("Valid strike?", key=state_keys["valid_strike_key"])
                st.selectbox(
                    "Corrected category",
                    options=CORRECTED_CATEGORIES,
                    key=state_keys["corrected_category_key"],
                )
                st.selectbox(
                    "Corrected strike type (type to search)",
                    options=CORRECTED_STRIKE_TYPES,
                    format_func=_format_strike_type_option,
                    key=state_keys["corrected_strike_type_key"],
                )
                st.text_area(
                    "Notes",
                    key=state_keys["notes_key"],
                    height=100,
                )

            with st.expander("Clip Debug Details"):
                st.caption(f"Clip path: `{clip_path.as_posix()}`")
                if clip_path.exists():
                    st.caption(f"File size: {clip_path.stat().st_size} bytes")
                st.caption(f"Duration: {float(clip.get('duration', 0.0)):.4f} seconds")
                st.caption(f"Codec source: `{clip.get('codec_source', 'unknown')}`")
                st.caption(
                    f"Validation status: `{clip.get('validation_status', 'unknown')}`"
                )
                st.caption(f"FFmpeg available: `{clip.get('ffmpeg_available', False)}`")
                st.caption(f"FFmpeg path: `{clip.get('ffmpeg_path', '')}`")
                if suggested_strike_type:
                    st.caption(f"Model suggestion stored: `{suggested_strike_type}`")
                if clip.get("ffmpeg_command"):
                    st.caption("FFmpeg command:")
                    st.code(str(clip.get("ffmpeg_command")), language="text")
                if clip.get("ffmpeg_stderr"):
                    st.caption("FFmpeg stderr:")
                    st.code(str(clip.get("ffmpeg_stderr")), language="text")


def render_filtered_clip_review(filtered_labels_df: pd.DataFrame) -> None:
    """Replay filtered labeled clips for dataset inspection."""
    st.subheader("Review Filtered Clips")

    if filtered_labels_df.empty:
        st.info("No clips match the current filters.")
        return

    clip_selection_options = [
        (
            f"{label['video_filename']} | Event {int(label['event_id'])} | "
            f"{label['corrected_category']} / {label['corrected_strike_type']}"
        )
        for label in filtered_labels_df.to_dict(orient="records")
    ]
    option_to_row_index = {
        option: index for index, option in enumerate(clip_selection_options)
    }
    selected_clip_options = st.multiselect(
        "Select filtered clips to replay",
        options=clip_selection_options,
        default=[],
        help="No filtered clips are shown until you select them.",
        key="dataset_review_selected_clips",
    )

    if not selected_clip_options:
        st.info("Select one or more filtered clips to replay.")
        return

    selected_row_indexes = [
        option_to_row_index[selected_option] for selected_option in selected_clip_options
    ]
    selected_labels_df = filtered_labels_df.iloc[selected_row_indexes].copy()

    for label in selected_labels_df.to_dict(orient="records"):
        clip_path = Path(str(label["clip_path"]))
        with st.container(border=True):
            st.markdown(
                f"**{label['video_filename']} | Event {int(label['event_id'])} | "
                f"{label['corrected_category']} / {label['corrected_strike_type']}**"
            )
            if not clip_path.exists():
                st.warning(f"Clip file does not exist: {clip_path.as_posix()}")
            elif clip_path.stat().st_size <= 0:
                st.warning(f"Clip file is empty: {clip_path.as_posix()}")
            else:
                with clip_path.open("rb") as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes, format="video/mp4")

            st.caption(
                f"Predicted: {label['predicted_category']} | "
                f"Corrected: {label['corrected_category']} | "
                f"Strike type: {label['corrected_strike_type']} | "
                f"Valid strike: {bool(label['is_valid_strike'])}"
            )


def render_dataset_dashboard() -> None:
    """Render dataset-level label review, filtering, replay, and clean export."""
    st.divider()
    st.subheader("Dataset & Label Review")

    labels_csv_path = build_event_labels_csv_path()
    if not labels_csv_path.exists():
        st.info("No saved labels found yet. Review clips and save labels to build the dataset.")
        return

    try:
        labels_df = pd.read_csv(labels_csv_path)
    except Exception as exc:
        st.error(f"Unable to load labels dataset: {exc}")
        return

    if labels_df.empty:
        st.info("The labels file exists but does not contain any rows yet.")
        return

    labels_df["is_valid_strike"] = _normalize_bool_series(labels_df["is_valid_strike"])
    labels_df["predicted_confidence_score"] = pd.to_numeric(
        labels_df["predicted_confidence_score"],
        errors="coerce",
    ).fillna(0.0)
    labels_df["corrected_category"] = labels_df["corrected_category"].fillna("unknown")
    labels_df["corrected_strike_type"] = labels_df["corrected_strike_type"].fillna("unknown")
    labels_df["coarse_strike_type"] = labels_df["corrected_strike_type"].apply(
        get_coarse_strike_type
    )
    labels_df["video_filename"] = labels_df["video_filename"].fillna("unknown_video")
    if "analysis_mode" not in labels_df.columns:
        labels_df["analysis_mode"] = "single_athlete_training"
    labels_df["predicted_category"] = labels_df["predicted_category"].fillna("unknown")
    labels_df["predicted_primary_limb"] = labels_df["predicted_primary_limb"].fillna("unknown")

    total_labeled_clips = int(len(labels_df))
    valid_strikes = int(labels_df["is_valid_strike"].sum())
    false_positives = int((labels_df["corrected_category"] == "false_positive").sum())
    unknown_labels = int(
        (
            (labels_df["corrected_category"] == "unknown")
            | (labels_df["corrected_strike_type"] == "unknown")
        ).sum()
    )
    unique_source_videos = int(labels_df["video_filename"].nunique())
    corrected_strike_type_count = int(
        labels_df.loc[labels_df["corrected_strike_type"] != "unknown", "corrected_strike_type"]
        .nunique()
    )

    metric_cols = st.columns(6)
    metric_cols[0].metric("Total Labeled Clips", total_labeled_clips)
    metric_cols[1].metric("Valid Strikes", valid_strikes)
    metric_cols[2].metric("False Positives", false_positives)
    metric_cols[3].metric("Unknown Labels", unknown_labels)
    metric_cols[4].metric("Unique Source Videos", unique_source_videos)
    metric_cols[5].metric("Corrected Strike Types", corrected_strike_type_count)

    category_chart = (
        alt.Chart(labels_df)
        .mark_bar()
        .encode(
            x=alt.X("count():Q", title="Count"),
            y=alt.Y("corrected_category:N", sort="-x", title="Corrected Category"),
            tooltip=["corrected_category", "count():Q"],
        )
        .properties(title="Corrected Category Distribution")
    )
    strike_type_chart = (
        alt.Chart(labels_df)
        .mark_bar()
        .encode(
            x=alt.X("count():Q", title="Count"),
            y=alt.Y("corrected_strike_type:N", sort="-x", title="Corrected Strike Type"),
            tooltip=["corrected_strike_type", "count():Q"],
        )
        .properties(title="Corrected Strike Type Distribution")
    )
    predicted_vs_corrected_chart = (
        alt.Chart(labels_df)
        .mark_rect()
        .encode(
            x=alt.X("predicted_category:N", title="Predicted Category"),
            y=alt.Y("corrected_category:N", title="Corrected Category"),
            color=alt.Color("count():Q", title="Count"),
            tooltip=["predicted_category", "corrected_category", "count():Q"],
        )
        .properties(title="Predicted vs Corrected Category")
    )
    primary_limb_chart = (
        alt.Chart(labels_df)
        .mark_bar()
        .encode(
            x=alt.X("count():Q", title="Count"),
            y=alt.Y("predicted_primary_limb:N", sort="-x", title="Predicted Primary Limb"),
            tooltip=["predicted_primary_limb", "count():Q"],
        )
        .properties(title="Predicted Primary Limb Distribution")
    )
    source_video_chart = (
        alt.Chart(labels_df)
        .mark_bar()
        .encode(
            x=alt.X("count():Q", title="Count"),
            y=alt.Y("video_filename:N", sort="-x", title="Source Video"),
            tooltip=["video_filename", "count():Q"],
        )
        .properties(title="Labels by Source Video")
    )

    chart_col_left, chart_col_right = st.columns(2)
    with chart_col_left:
        st.altair_chart(category_chart, use_container_width=True)
        st.altair_chart(predicted_vs_corrected_chart, use_container_width=True)
        st.altair_chart(source_video_chart, use_container_width=True)
    with chart_col_right:
        st.altair_chart(strike_type_chart, use_container_width=True)
        st.altair_chart(primary_limb_chart, use_container_width=True)

    st.write("Filters")
    filter_col_1, filter_col_2, filter_col_3 = st.columns(3)
    with filter_col_1:
        selected_videos = st.multiselect(
            "Source video",
            options=sorted(labels_df["video_filename"].unique().tolist()),
            default=[],
        )
        selected_categories = st.multiselect(
            "Corrected category",
            options=sorted(labels_df["corrected_category"].unique().tolist()),
            default=[],
        )
    with filter_col_2:
        selected_strike_types = st.multiselect(
            "Corrected strike type",
            options=sorted(labels_df["corrected_strike_type"].unique().tolist()),
            default=[],
        )
        valid_strike_filter = st.selectbox(
            "Valid strike",
            options=["all", "true", "false"],
            index=0,
        )
    with filter_col_3:
        min_confidence = float(labels_df["predicted_confidence_score"].min())
        max_confidence = float(labels_df["predicted_confidence_score"].max())
        confidence_range = st.slider(
            "Confidence score range",
            min_value=min_confidence,
            max_value=max_confidence if max_confidence > min_confidence else min_confidence + 0.001,
            value=(
                min_confidence,
                max_confidence if max_confidence > min_confidence else min_confidence + 0.001,
            ),
            step=0.01,
        )

    filtered_labels_df = labels_df.copy()
    if selected_videos:
        filtered_labels_df = filtered_labels_df[
            filtered_labels_df["video_filename"].isin(selected_videos)
        ]
    if selected_categories:
        filtered_labels_df = filtered_labels_df[
            filtered_labels_df["corrected_category"].isin(selected_categories)
        ]
    if selected_strike_types:
        filtered_labels_df = filtered_labels_df[
            filtered_labels_df["corrected_strike_type"].isin(selected_strike_types)
        ]
    if valid_strike_filter != "all":
        filtered_labels_df = filtered_labels_df[
            filtered_labels_df["is_valid_strike"] == (valid_strike_filter == "true")
        ]
    filtered_labels_df = filtered_labels_df[
        filtered_labels_df["predicted_confidence_score"].between(
            confidence_range[0],
            confidence_range[1],
        )
    ]

    st.write("Filtered labels table")
    st.dataframe(filtered_labels_df, use_container_width=True)

    render_filtered_clip_review(filtered_labels_df)

    if st.button("Export Clean Dataset"):
        clean_dataset_df = labels_df[
            (labels_df["is_valid_strike"])
            & (labels_df["corrected_category"] != "unknown")
            & (labels_df["corrected_strike_type"] != "unknown")
        ].copy()
        if "analysis_mode" not in clean_dataset_df.columns:
            clean_dataset_df["analysis_mode"] = "single_athlete_training"
        clean_dataset_df["coarse_strike_type"] = clean_dataset_df[
            "corrected_strike_type"
        ].apply(get_coarse_strike_type)
        clean_dataset_path = build_clean_labeled_events_csv_path()
        clean_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        clean_dataset_df.to_csv(clean_dataset_path, index=False)
        st.success(
            f"Exported {len(clean_dataset_df)} rows to `{clean_dataset_path.as_posix()}`"
        )

        phase19_feature_dataset_path = build_phase19_feature_dataset_csv_path()
        try:
            feature_dataset_df, feature_warnings = build_feature_dataset(
                clean_labels_path=clean_dataset_path,
                processed_dir="data/processed",
                output_path=phase19_feature_dataset_path,
            )
        except ValueError as exc:
            st.error(f"Clean dataset exported, but Phase 19 feature rebuild failed: {exc}")
            return
        except Exception as exc:
            st.error(f"Clean dataset exported, but Phase 19 feature rebuild failed: {exc}")
            return

        st.session_state["ml_feature_dataset_df"] = feature_dataset_df
        st.session_state["ml_feature_dataset_warnings"] = feature_warnings
        st.session_state["ml_feature_dataset_path"] = str(phase19_feature_dataset_path)
        st.success(
            f"Rebuilt Phase 19 feature dataset with {len(feature_dataset_df)} rows "
            f"at `{phase19_feature_dataset_path.as_posix()}`"
        )
        if feature_warnings:
            with st.expander("Feature Rebuild Warnings"):
                for warning in feature_warnings:
                    st.warning(warning)


def render_ml_feature_dataset_section() -> None:
    """Render the ML-ready feature dataset builder and QA view."""
    st.divider()
    st.subheader("Phase 19 Feature Dataset")
    st.caption("Build numeric model-ready features from clean labeled events into the locked Phase 19 path.")

    if st.button("Build Feature Dataset"):
        clean_labels_path = build_clean_labeled_events_csv_path()
        ml_feature_dataset_path = build_phase19_feature_dataset_csv_path()

        try:
            feature_dataset_df, feature_warnings = build_feature_dataset(
                clean_labels_path=clean_labels_path,
                processed_dir="data/processed",
                output_path=ml_feature_dataset_path,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Building ML feature dataset failed: {exc}")
            return

        st.session_state["ml_feature_dataset_df"] = feature_dataset_df
        st.session_state["ml_feature_dataset_warnings"] = feature_warnings
        st.session_state["ml_feature_dataset_path"] = str(ml_feature_dataset_path)

        st.success(
            f"Built feature dataset with {len(feature_dataset_df)} rows and "
            f"{len(feature_dataset_df.columns)} columns at `{ml_feature_dataset_path.as_posix()}`"
        )

    feature_dataset_df = st.session_state.get("ml_feature_dataset_df")
    feature_warnings = st.session_state.get("ml_feature_dataset_warnings", [])
    feature_dataset_path = st.session_state.get("ml_feature_dataset_path")

    if isinstance(feature_dataset_df, pd.DataFrame):
        st.write(
            f"Dataset summary: {len(feature_dataset_df)} rows, {len(feature_dataset_df.columns)} columns"
        )
        if feature_dataset_path:
            st.caption(f"Saved to `{feature_dataset_path}`")

        if feature_warnings:
            with st.expander("Feature Build Warnings"):
                for warning in feature_warnings:
                    st.warning(warning)

        if feature_dataset_df.empty:
            st.warning("The feature dataset was created but contains no usable rows.")
            return

        st.write("First 20 rows")
        st.dataframe(feature_dataset_df.head(20), use_container_width=True)

        strike_type_distribution = (
            feature_dataset_df["corrected_strike_type"].value_counts().reset_index()
        )
        strike_type_distribution.columns = ["corrected_strike_type", "count"]
        strike_type_chart = (
            alt.Chart(strike_type_distribution)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("corrected_strike_type:N", sort="-x", title="Corrected Strike Type"),
                tooltip=["corrected_strike_type", "count"],
            )
            .properties(title="Corrected Strike Type Distribution")
        )
        st.altair_chart(strike_type_chart, use_container_width=True)

        small_classes = strike_type_distribution[strike_type_distribution["count"] < 5]
        if not small_classes.empty:
            class_names = ", ".join(small_classes["corrected_strike_type"].tolist())
            st.warning(
                f"Some classes have fewer than 5 examples: {class_names}"
            )


def render_baseline_model_training_section() -> None:
    """Render the Phase 19 fine strike model training and validation dashboard."""
    st.divider()
    st.subheader("Fine Strike Model")
    st.caption("Train and validate the single-athlete fine strike classifier.")

    clean_dataset_path = build_clean_labeled_events_csv_path()
    feature_dataset_path = build_phase19_feature_dataset_csv_path()
    selected_features_path = build_selected_fine_features_csv_path()
    model_metadata_path = build_baseline_model_metadata_path()
    model_metadata = _safe_json_dict(model_metadata_path)

    clean_dataset_rows = _safe_csv_row_count(clean_dataset_path)
    feature_dataset_rows = _safe_csv_row_count(feature_dataset_path)
    selected_feature_count = _safe_selected_feature_count(selected_features_path)
    feature_dataset_stale = _is_feature_dataset_stale(
        clean_dataset_path,
        feature_dataset_path,
    )
    model_stale = _is_model_stale(
        build_baseline_model_path(),
        model_metadata_path,
        feature_dataset_path,
        selected_features_path,
    )

    status_cols = st.columns(4)
    status_cols[0].metric("Clean Rows", _format_project_state_value(clean_dataset_rows))
    status_cols[1].metric(
        "Training Rows",
        _format_project_state_value(feature_dataset_rows),
    )
    status_cols[2].metric(
        "Selected Features",
        _format_project_state_value(selected_feature_count),
    )
    status_cols[3].metric(
        "Model Rows",
        _format_project_state_value(model_metadata.get("dataset_rows")),
    )

    if feature_dataset_stale:
        st.warning(
            "The Phase 19 feature dataset is stale relative to clean labels. "
            "Export Clean Dataset again to rebuild training features automatically."
        )
    elif feature_dataset_stale is False:
        st.success("Phase 19 feature dataset is current with the clean labels.")

    if model_stale:
        st.warning("The saved fine strike model is stale relative to the current training data.")
    elif model_stale is False:
        st.success("The saved fine strike model is current with the Phase 19 training data.")

    if not feature_dataset_path.exists():
        st.info(
            f"Phase 19 training dataset not found: `{feature_dataset_path.as_posix()}`"
        )
        return
    if not selected_features_path.exists():
        st.info(
            f"Selected feature list not found: `{selected_features_path.as_posix()}`"
        )
        return

    try:
        feature_dataset_df = pd.read_csv(feature_dataset_path)
    except Exception as exc:
        st.error(f"Unable to load ML feature dataset: {exc}")
        return

    if feature_dataset_df.empty:
        st.warning("The ML feature dataset exists but contains no rows.")
        return

    target_column = "corrected_strike_type"
    model_output_path = build_baseline_model_path()
    metrics_output_path = build_model_evaluation_csv_path()

    if target_column not in feature_dataset_df.columns:
        st.warning(
            f"The ML feature dataset does not include `{target_column}` yet. Rebuild the feature dataset first."
        )
        return

    training_df = feature_dataset_df.copy()
    training_df[target_column] = training_df[target_column].astype(str).str.strip()
    training_df = training_df[
        training_df[target_column].notna()
        & (training_df[target_column] != "")
        & (training_df[target_column].str.lower() != "unknown")
        & (training_df[target_column].str.lower() != "nan")
    ].copy()

    st.metric("Dataset Row Count", int(len(training_df)))
    st.caption(f"Training source: `{feature_dataset_path.as_posix()}`")
    st.caption(f"Selected features: `{selected_features_path.as_posix()}`")

    if training_df.empty:
        st.warning(f"No rows remain after filtering out null or unknown values for `{target_column}`.")
        return

    class_distribution_df = training_df[target_column].value_counts().reset_index()
    class_distribution_df.columns = [target_column, "count"]
    st.write("Class distribution")
    st.dataframe(class_distribution_df, use_container_width=True)

    class_distribution_chart = (
        alt.Chart(class_distribution_df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y(f"{target_column}:N", sort="-x", title="Target Label"),
            tooltip=[target_column, "count"],
        )
        .properties(title=f"Training Class Distribution: {target_column}")
    )
    st.altair_chart(class_distribution_chart, use_container_width=True)

    small_classes = class_distribution_df[class_distribution_df["count"] < 5]
    if not small_classes.empty:
        class_names = ", ".join(small_classes[target_column].tolist())
        st.warning(f"Some classes have fewer than 5 examples: {class_names}")

    if st.button("Train Fine Strike Model"):
        try:
            training_results = train_baseline_model(
                dataset_path=feature_dataset_path,
                target_column=target_column,
                model_output_path=model_output_path,
                metrics_output_path=metrics_output_path,
                selected_features_path=selected_features_path,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Fine strike model training failed: {exc}")
            return

        st.session_state[f"baseline_model_training_results_{target_column}"] = training_results
        st.success(
            f"Fine strike model saved to `{Path(training_results['model_path']).as_posix()}` "
            f"with metadata at `{Path(training_results['metadata_path']).as_posix()}`"
        )

    training_results = st.session_state.get(
        f"baseline_model_training_results_{target_column}"
    )
    if not isinstance(training_results, dict):
        return

    for warning in training_results.get("warnings", []):
        st.warning(warning)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Training Rows", int(training_results["training_rows"]))
    summary_cols[1].metric("Test Rows", int(training_results["test_rows"]))
    summary_cols[2].metric("Numeric Features", int(training_results["feature_count"]))
    summary_cols[3].metric(
        "Split Strategy",
        "Stratified" if training_results["stratified_split_used"] else "Simple",
    )
    st.caption(f"Evaluation target: `{training_results['target_column']}`")

    st.write(f"Classification report: {training_results['target_column']}")
    st.dataframe(training_results["classification_report_df"], use_container_width=True)

    st.write(f"Confusion matrix: {training_results['target_column']}")
    confusion_matrix_df = training_results["confusion_matrix_df"]
    st.dataframe(confusion_matrix_df, use_container_width=True)

    confusion_matrix_long_df = confusion_matrix_df.melt(
        id_vars="actual_label",
        var_name="predicted_label",
        value_name="count",
    )
    confusion_matrix_chart = (
        alt.Chart(confusion_matrix_long_df)
        .mark_rect()
        .encode(
            x=alt.X("predicted_label:N", title="Predicted Label"),
            y=alt.Y("actual_label:N", title="Actual Label"),
            color=alt.Color("count:Q", title="Count"),
            tooltip=["actual_label", "predicted_label", "count"],
        )
        .properties(title=f"Confusion Matrix Heatmap: {training_results['target_column']}")
    )
    st.altair_chart(confusion_matrix_chart, use_container_width=True)

    st.write("Top 20 feature importances")
    st.dataframe(training_results["top_feature_importances_df"], use_container_width=True)

    feature_importance_chart = (
        alt.Chart(training_results["top_feature_importances_df"])
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature_name:N", sort="-x", title="Feature"),
            tooltip=["feature_name", "importance"],
        )
        .properties(title="Top 20 Feature Importances")
    )
    st.altair_chart(feature_importance_chart, use_container_width=True)


def render_video_holdout_validation_section() -> None:
    """Render read-only leave-one-video-out validation for the locked feature set."""
    st.divider()
    st.subheader("Video Holdout Validation")
    st.caption(
        "Train temporary models on all videos except one, then validate on the held-out video. "
        "This does not overwrite the active fine strike model."
    )

    feature_dataset_path = build_phase19_feature_dataset_csv_path()
    selected_features_path = build_selected_fine_features_csv_path()
    summary_output_path = build_video_holdout_summary_csv_path()
    per_class_output_path = build_video_holdout_per_class_csv_path()
    confusion_output_path = build_video_holdout_confusion_csv_path()

    status_cols = st.columns(3)
    status_cols[0].metric(
        "Feature Rows",
        _format_project_state_value(_safe_csv_row_count(feature_dataset_path)),
    )
    status_cols[1].metric(
        "Selected Features",
        _format_project_state_value(_safe_selected_feature_count(selected_features_path)),
    )
    status_cols[2].metric(
        "Previous Runs",
        "Available" if summary_output_path.exists() else "Not yet run",
    )

    if not feature_dataset_path.exists():
        st.info("Build the Phase 19 feature dataset before running video holdout validation.")
        return
    if not selected_features_path.exists():
        st.info("Selected feature list is missing.")
        return

    if st.button("Run Video Holdout Validation"):
        try:
            validation_results = validate_by_heldout_video(
                dataset_path=feature_dataset_path,
                selected_features_path=selected_features_path,
                summary_output_path=summary_output_path,
                per_class_output_path=per_class_output_path,
                confusion_output_path=confusion_output_path,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Video holdout validation failed: {exc}")
            return

        st.session_state["video_holdout_validation_results"] = validation_results
        st.success(
            f"Saved video holdout validation to `{summary_output_path.as_posix()}`"
        )

    validation_results = st.session_state.get("video_holdout_validation_results")
    if isinstance(validation_results, dict):
        summary_df = validation_results["summary_df"]
        per_class_df = validation_results["per_class_df"]
        confusion_df = validation_results["confusion_df"]
        warnings = validation_results.get("warnings", [])
    elif summary_output_path.exists():
        try:
            summary_df = pd.read_csv(summary_output_path)
            per_class_df = pd.read_csv(per_class_output_path)
            confusion_df = pd.read_csv(confusion_output_path)
            warnings = []
        except Exception as exc:
            st.warning(f"Unable to load previous validation outputs: {exc}")
            return
    else:
        return

    if summary_df.empty:
        st.warning("Video holdout validation produced no summary rows.")
        return

    for warning in warnings:
        st.warning(warning)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Held-Out Videos", int(len(summary_df)))
    metric_cols[1].metric("Mean Accuracy", f"{float(summary_df['accuracy'].mean()):.3f}")
    metric_cols[2].metric("Mean Macro F1", f"{float(summary_df['macro_f1'].mean()):.3f}")
    metric_cols[3].metric(
        "Mean Weighted F1",
        f"{float(summary_df['weighted_f1'].mean()):.3f}",
    )

    st.write("Per-video validation summary")
    st.dataframe(
        summary_df.sort_values("weighted_f1", ascending=True),
        use_container_width=True,
    )

    worst_videos_df = summary_df.sort_values("weighted_f1", ascending=True).head(5)
    worst_chart = (
        alt.Chart(worst_videos_df)
        .mark_bar()
        .encode(
            x=alt.X("weighted_f1:Q", title="Weighted F1"),
            y=alt.Y("heldout_video:N", sort="x", title="Held-Out Video"),
            tooltip=["heldout_video", "test_rows", "accuracy", "macro_f1", "weighted_f1"],
        )
        .properties(title="Lowest Held-Out Video Scores")
    )
    st.altair_chart(worst_chart, use_container_width=True)

    if not per_class_df.empty:
        st.write("Per-video class recall")
        st.dataframe(
            per_class_df.sort_values(["recall", "support"], ascending=[True, False]),
            use_container_width=True,
        )

    if not confusion_df.empty:
        st.write("Non-zero confusion counts")
        st.dataframe(confusion_df, use_container_width=True)

    st.caption(f"Summary CSV: `{summary_output_path.as_posix()}`")
    st.caption(f"Per-class CSV: `{per_class_output_path.as_posix()}`")
    st.caption(f"Confusion CSV: `{confusion_output_path.as_posix()}`")


def _format_project_state_value(value: object | None) -> str:
    """Format optional project-state values for consistent sidebar display."""
    if value is None or value == "":
        return "Not yet created"
    return str(value)


def _format_status_flag(value: bool | None) -> str:
    """Format freshness flags for compact status display."""
    if value is None:
        return "Unknown"
    return "Stale" if value else "Current"


def _safe_csv_row_count(csv_path: Path) -> int | None:
    """Return CSV row count when the file exists and is readable."""
    if not csv_path.exists():
        return None
    try:
        return int(len(pd.read_csv(csv_path)))
    except Exception:
        return None


def _safe_selected_feature_count(csv_path: Path) -> int | None:
    """Return selected feature count when the feature list exists and is readable."""
    if not csv_path.exists():
        return None
    try:
        feature_df = pd.read_csv(csv_path)
    except Exception:
        return None
    if feature_df.empty:
        return 0
    feature_column = "feature_name" if "feature_name" in feature_df.columns else feature_df.columns[0]
    return int(
        feature_df[feature_column]
        .astype(str)
        .str.strip()
        .loc[lambda series: (series != "") & (series.str.lower() != "nan")]
        .nunique()
    )


def _safe_file_timestamp(file_path: Path) -> str | None:
    """Return the modified timestamp for a file in local time."""
    if not file_path.exists():
        return None
    return datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _safe_json_dict(json_path: Path) -> dict[str, object]:
    """Load a JSON object when present and readable."""
    if not json_path.exists():
        return {}
    try:
        loaded_value = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded_value if isinstance(loaded_value, dict) else {}


def _is_feature_dataset_stale(clean_dataset_path: Path, feature_dataset_path: Path) -> bool | None:
    """Return whether Phase 19 features are stale relative to clean labels."""
    if not clean_dataset_path.exists():
        return None
    if not feature_dataset_path.exists():
        return True

    clean_rows = _safe_csv_row_count(clean_dataset_path)
    feature_rows = _safe_csv_row_count(feature_dataset_path)
    if clean_rows is None or feature_rows is None:
        return None
    if clean_rows != feature_rows:
        return True
    return feature_dataset_path.stat().st_mtime < clean_dataset_path.stat().st_mtime


def _is_model_stale(
    model_path: Path,
    metadata_path: Path,
    feature_dataset_path: Path,
    selected_features_path: Path,
) -> bool | None:
    """Return whether the saved model is stale relative to Phase 19 training inputs."""
    if not feature_dataset_path.exists() or not selected_features_path.exists():
        return None
    if not model_path.exists() or not metadata_path.exists():
        return True

    metadata = _safe_json_dict(metadata_path)
    feature_rows = _safe_csv_row_count(feature_dataset_path)
    selected_feature_count = _safe_selected_feature_count(selected_features_path)
    if feature_rows is None or selected_feature_count is None or not metadata:
        return None

    if metadata.get("dataset_rows") != feature_rows:
        return True
    if metadata.get("feature_count") != selected_feature_count:
        return True
    latest_input_mtime = max(feature_dataset_path.stat().st_mtime, selected_features_path.stat().st_mtime)
    return model_path.stat().st_mtime < latest_input_mtime


def _list_uploaded_videos() -> list[Path]:
    """List reusable uploaded videos from the uploads directory."""
    return sorted(
        [path for path in UPLOADS_DIR.glob("*.mp4") if path.is_file()],
        key=lambda path: path.name.lower(),
    )


def _get_selected_video_path() -> Path | None:
    """Resolve the currently selected video from session state."""
    selected_path = st.session_state.get("selected_video_path")
    if not selected_path:
        return None
    video_path = Path(str(selected_path))
    if not video_path.exists():
        return None
    return video_path


def get_project_state_summary() -> dict[str, object | None]:
    """Collect project state values for sidebar and dashboard views."""
    labels_csv_path = build_event_labels_csv_path()
    clean_dataset_path = build_clean_labeled_events_csv_path()
    feature_dataset_path = build_phase19_feature_dataset_csv_path()
    selected_features_path = build_selected_fine_features_csv_path()
    fine_model_path = build_baseline_model_path()
    fine_model_metadata_path = build_baseline_model_metadata_path()
    model_metadata = _safe_json_dict(fine_model_metadata_path)

    total_labeled_clips: int | None = None
    valid_strikes: int | None = None
    unique_videos: int | None = None

    if labels_csv_path.exists():
        try:
            labels_df = pd.read_csv(labels_csv_path)
            total_labeled_clips = int(len(labels_df))
            labels_df["is_valid_strike"] = _normalize_bool_series(labels_df["is_valid_strike"])
            valid_strikes = int(labels_df["is_valid_strike"].sum())
            unique_videos = int(labels_df["video_filename"].fillna("unknown").nunique())
        except Exception:
            total_labeled_clips = None
            valid_strikes = None
            unique_videos = None

    selected_video_path = _get_selected_video_path()
    return {
        "selected_video_name": selected_video_path.name if selected_video_path else None,
        "total_labeled_clips": total_labeled_clips,
        "valid_strikes": valid_strikes,
        "unique_videos": unique_videos,
        "clean_dataset_rows": _safe_csv_row_count(clean_dataset_path),
        "feature_dataset_rows": _safe_csv_row_count(feature_dataset_path),
        "selected_feature_count": _safe_selected_feature_count(selected_features_path),
        "feature_dataset_stale": _is_feature_dataset_stale(
            clean_dataset_path,
            feature_dataset_path,
        ),
        "model_stale": _is_model_stale(
            fine_model_path,
            fine_model_metadata_path,
            feature_dataset_path,
            selected_features_path,
        ),
        "fine_model_exists": fine_model_path.exists(),
        "fine_model_last_trained": _safe_file_timestamp(fine_model_path),
        "fine_model_file_path": fine_model_path.as_posix() if fine_model_path.exists() else None,
        "fine_model_dataset_rows": model_metadata.get("dataset_rows"),
        "fine_model_feature_count": model_metadata.get("feature_count"),
        "fine_model_metadata_path": fine_model_metadata_path.as_posix()
        if fine_model_metadata_path.exists()
        else None,
    }


def render_project_state_panel() -> None:
    """Render the sidebar project state panel."""
    project_state = get_project_state_summary()

    with st.sidebar.expander("Project State", expanded=False):
        st.caption("Current project artifacts and active context.")
        st.write(
            f"Last selected video: `{_format_project_state_value(project_state['selected_video_name'])}`"
        )
        st.write(
            f"Total labeled clips: `{_format_project_state_value(project_state['total_labeled_clips'])}`"
        )
        st.write(
            f"Valid strikes: `{_format_project_state_value(project_state['valid_strikes'])}`"
        )
        st.write(
            f"Unique videos in dataset: `{_format_project_state_value(project_state['unique_videos'])}`"
        )
        st.write(
            f"Clean dataset rows: `{_format_project_state_value(project_state['clean_dataset_rows'])}`"
        )
        st.write(
            f"ML feature rows: `{_format_project_state_value(project_state['feature_dataset_rows'])}`"
        )
        st.write(
            f"Feature dataset freshness: `{_format_status_flag(project_state['feature_dataset_stale'])}`"
        )
        st.write(
            f"Selected features: `{_format_project_state_value(project_state['selected_feature_count'])}`"
        )
        st.write(
            f"Fine model exists: `{_format_project_state_value(project_state['fine_model_exists'])}`"
        )
        st.write(
            f"Fine model trained: `{_format_project_state_value(project_state['fine_model_last_trained'])}`"
        )
        st.write(
            f"Fine model freshness: `{_format_status_flag(project_state['model_stale'])}`"
        )
        st.write(
            f"Fine model rows: `{_format_project_state_value(project_state['fine_model_dataset_rows'])}`"
        )
        st.write(
            f"Fine model path: `{_format_project_state_value(project_state['fine_model_file_path'])}`"
        )


def render_sidebar_navigation() -> str:
    """Render top-level navigation as the primary sidebar control."""
    options = [
        "Dashboard",
        "Video Processing",
        "Event Detection",
        "Clip Review & Labeling",
        "Dataset & Label Review",
        "Fine Strike Model",
        "Video Holdout Validation",
    ]
    pending_section = st.session_state.pop("pending_section", None)
    if pending_section in options:
        st.session_state["current_section"] = pending_section

    current_section = st.session_state.get("current_section", "Dashboard")
    default_index = options.index(current_section) if current_section in options else 0

    st.sidebar.subheader("Navigation")
    selected_section = st.sidebar.radio(
        "Section",
        options=options,
        index=default_index,
    )
    st.session_state["current_section"] = selected_section
    return selected_section


def render_video_selector() -> tuple[Path | None, str]:
    """Render reusable video selection controls in the sidebar."""
    with st.sidebar.expander("Video Context", expanded=True):
        video_source = st.radio(
            "Choose video source",
            options=["Select Existing Video", "Upload New Video"],
            index=0,
        )
        analysis_mode = "single_athlete_training"

        if video_source == "Upload New Video":
            uploaded_file = st.file_uploader(
                "Upload an MP4 video",
                type=["mp4"],
                help="Only MP4 files are supported in the current phase.",
                key="sidebar_video_uploader",
            )
            if uploaded_file is not None:
                if not is_valid_video_file(uploaded_file.name):
                    st.error("Invalid file type. Please upload an MP4 video.")
                else:
                    upload_signature = f"{uploaded_file.name}:{uploaded_file.size}"
                    if st.session_state.get("uploaded_video_signature") != upload_signature:
                        try:
                            saved_path = save_uploaded_video(uploaded_file)
                        except ValueError as exc:
                            st.error(str(exc))
                        except Exception as exc:
                            st.error(f"Saving uploaded video failed: {exc}")
                        else:
                            st.session_state["uploaded_video_signature"] = upload_signature
                            st.session_state["selected_video_path"] = str(saved_path)
                            st.success(f"Selected `{saved_path.name}`")
        else:
            available_videos = _list_uploaded_videos()
            if not available_videos:
                st.info("No saved uploads found yet.")
                st.session_state["selected_video_path"] = None
            else:
                video_names = [path.name for path in available_videos]
                current_video_path = _get_selected_video_path()
                default_index = 0
                if current_video_path and current_video_path.name in video_names:
                    default_index = video_names.index(current_video_path.name)
                selected_video_name = st.selectbox(
                    "Select existing video",
                    options=video_names,
                    index=default_index,
                )
                selected_video_path = next(
                    path for path in available_videos if path.name == selected_video_name
                )
                st.session_state["selected_video_path"] = str(selected_video_path)

        selected_video_path = _get_selected_video_path()
        if selected_video_path is not None:
            st.caption(f"Active video: `{selected_video_path.name}`")

    return selected_video_path, analysis_mode


def render_video_required_message() -> None:
    """Display a standard message for video-dependent workflows."""
    st.info("Please upload or select a video to use this section.")


def build_model_suggestions_for_clips(
    clips_df: pd.DataFrame,
    selected_video_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach model-assisted strike-type suggestions to exported clips when possible."""
    model_path = build_baseline_model_path()
    if clips_df.empty or not model_path.exists():
        return clips_df.copy(), []

    pose_csv_path = build_processed_pose_csv_path(selected_video_path)
    if not pose_csv_path.exists():
        return clips_df.copy(), [f"Pose CSV not found for model suggestions: {pose_csv_path.name}"]

    try:
        pose_df = pd.read_csv(pose_csv_path)
    except Exception as exc:
        return clips_df.copy(), [f"Unable to load pose CSV for model suggestions: {exc}"]

    enriched_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    for clip in clips_df.to_dict(orient="records"):
        enriched_clip = dict(clip)
        try:
            prediction = predict_event_strike_type(
                event_row=enriched_clip,
                pose_df=pose_df,
                model_path=model_path,
            )
        except Exception as exc:
            warnings.append(
                f"Model suggestion failed for event {int(clip['event_id'])}: {exc}"
            )
            prediction = {
                "suggested_strike_type": "",
                "prediction_confidence": None,
                "top_3_predictions_json": "",
                "warnings": [],
            }
        for warning in prediction.get("warnings", []):
            warnings.append(f"Event {int(clip['event_id'])}: {warning}")

        enriched_clip["model_suggested_strike_type"] = prediction.get(
            "suggested_strike_type", ""
        )
        enriched_clip["model_prediction_confidence"] = prediction.get(
            "prediction_confidence", None
        )
        enriched_clip["model_top_3_predictions"] = prediction.get(
            "top_3_predictions_json", ""
        )
        enriched_rows.append(enriched_clip)

    return pd.DataFrame(enriched_rows), warnings


def _go_to_section(section_name: str) -> None:
    """Switch the active section from a dashboard quick action."""
    st.session_state["pending_section"] = section_name
    st.rerun()


def render_dashboard_section() -> None:
    """Render the product-style landing dashboard."""
    project_state = get_project_state_summary()

    st.header("Dashboard")
    st.caption("Project overview, dataset readiness, and quick entry points into core workflows.")

    summary_cols = st.columns(5)
    summary_cols[0].metric(
        "Labeled Clips",
        _format_project_state_value(project_state["total_labeled_clips"]),
    )
    summary_cols[1].metric(
        "Valid Strikes",
        _format_project_state_value(project_state["valid_strikes"]),
    )
    summary_cols[2].metric(
        "Unique Videos",
        _format_project_state_value(project_state["unique_videos"]),
    )
    summary_cols[3].metric(
        "Clean Dataset Rows",
        _format_project_state_value(project_state["clean_dataset_rows"]),
    )
    summary_cols[4].metric(
        "ML Feature Rows",
        _format_project_state_value(project_state["feature_dataset_rows"]),
    )
    st.caption(
        "Training data freshness: "
        f"`{_format_status_flag(project_state['feature_dataset_stale'])}` | "
        "Model freshness: "
        f"`{_format_status_flag(project_state['model_stale'])}`"
    )

    st.divider()
    st.subheader("Model Status")
    model_status_cols = st.columns(1)
    model_status_cols[0].metric(
        "Fine Model",
        _format_project_state_value(project_state["fine_model_last_trained"]),
    )
    st.write(
        f"Fine model path: `{_format_project_state_value(project_state['fine_model_file_path'])}`"
    )
    st.write(
        f"Fine model rows: `{_format_project_state_value(project_state['fine_model_dataset_rows'])}`"
    )
    st.write(
        f"Fine model features: `{_format_project_state_value(project_state['fine_model_feature_count'])}`"
    )

    st.divider()
    st.subheader("Current Context")
    context_cols = st.columns(2)
    context_cols[0].write(
        f"Selected Video: `{_format_project_state_value(project_state['selected_video_name'])}`"
    )
    context_cols[1].write(
        f"Dataset Size: `{_format_project_state_value(project_state['feature_dataset_rows'])}` rows"
    )

    st.divider()
    st.subheader("Quick Actions")
    action_cols = st.columns(3)
    if action_cols[0].button("Go to Video Processing", use_container_width=True):
        _go_to_section("Video Processing")
    if action_cols[1].button("Go to Clip Review & Labeling", use_container_width=True):
        _go_to_section("Clip Review & Labeling")
    if action_cols[2].button("Go to Fine Strike Model", use_container_width=True):
        _go_to_section("Fine Strike Model")


def render_video_processing_section(selected_video_path: Path | None) -> None:
    """Render the standalone video preview and metadata view."""
    st.header("Video Processing")

    if selected_video_path is None:
        render_video_required_message()
        return

    st.caption(f"Selected video: `{selected_video_path.name}`")

    try:
        metadata = extract_video_metadata(selected_video_path)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Unable to load video metadata: {exc}")
        return

    st.subheader("Video Preview")
    st.video(str(selected_video_path))
    st.caption(f"Saved to `{selected_video_path.as_posix()}`")

    with st.expander("Video Metadata", expanded=False):
        metric_columns = st.columns(3)
        formatted_metadata = list(format_metadata(metadata).items())
        for index, (label, value) in enumerate(formatted_metadata):
            metric_columns[index % 3].metric(label=label, value=value)


def render_event_detection_section(
    selected_video_path: Path | None,
    analysis_mode: str,
) -> None:
    """Render pose extraction, overlay generation, and strike detection."""
    st.header("Event Detection")

    if selected_video_path is None:
        render_video_required_message()
        return

    st.caption(
        f"Selected video: `{selected_video_path.name}` | Analysis mode: `{analysis_mode}`"
    )
    st.subheader("Pose Extraction")

    sample_rate = st.number_input(
        "Sample every Nth frame",
        min_value=1,
        value=1,
        step=1,
        help="Use 1 to process every frame. Higher values process fewer frames.",
        key="event_detection_sample_rate",
    )

    if st.button("Extract Pose Data", type="primary"):
        output_csv_path = build_processed_pose_csv_path(selected_video_path)
        progress_bar = st.progress(0, text="Preparing pose extraction...")
        status_placeholder = st.empty()

        def update_progress(current_frame: int, total_frames: int) -> None:
            if total_frames <= 0:
                progress_bar.progress(0, text="Processing video frames...")
                return

            progress_value = min(int((current_frame / total_frames) * 100), 100)
            progress_bar.progress(
                progress_value,
                text=f"Processing frame {current_frame} of {total_frames}",
            )

        try:
            pose_df = extract_pose_from_video_with_progress(
                video_path=str(selected_video_path),
                output_csv_path=str(output_csv_path),
                sample_rate=int(sample_rate),
                analysis_mode=analysis_mode,
                progress_callback=update_progress,
            )
        except ValueError as exc:
            progress_bar.empty()
            status_placeholder.error(str(exc))
            return
        except Exception as exc:
            progress_bar.empty()
            status_placeholder.error(f"Pose extraction failed: {exc}")
            return

        progress_bar.progress(100, text="Pose extraction complete.")
        status_placeholder.success(f"Pose data saved to `{output_csv_path.as_posix()}`")
        st.write("First 20 rows of extracted pose data:")
        st.dataframe(pose_df.head(20), use_container_width=True)

    st.divider()
    st.subheader("Pose Overlay Video")

    if st.button("Generate Pose Overlay"):
        output_video_path = build_processed_overlay_video_path(selected_video_path)
        progress_bar = st.progress(0, text="Preparing pose overlay generation...")
        status_placeholder = st.empty()

        def update_overlay_progress(current_frame: int, total_frames: int) -> None:
            if total_frames <= 0:
                progress_bar.progress(0, text="Generating annotated video...")
                return

            progress_value = min(int((current_frame / total_frames) * 100), 100)
            progress_bar.progress(
                progress_value,
                text=f"Annotating frame {current_frame} of {total_frames}",
            )

        try:
            overlay_result = create_pose_overlay_video_with_diagnostics(
                input_video_path=str(selected_video_path),
                output_video_path=str(output_video_path),
                progress_callback=update_overlay_progress,
            )
        except ValueError as exc:
            progress_bar.empty()
            status_placeholder.error(str(exc))
            return
        except Exception as exc:
            progress_bar.empty()
            status_placeholder.error(f"Pose overlay generation failed: {exc}")
            return

        summary = overlay_result["summary"]
        annotated_video_path = str(overlay_result["output_video_path"])

        progress_bar.progress(100, text="Pose overlay video complete.")
        status_placeholder.success(f"Annotated video saved to `{output_video_path.as_posix()}`")

        summary_cols = st.columns(5)
        summary_cols[0].metric("Total Frames", int(summary["total_frames"]))
        summary_cols[1].metric("Pose Detected", int(summary["frames_with_pose_detected"]))
        summary_cols[2].metric("Pose Missing", int(summary["frames_with_pose_missing"]))
        summary_cols[3].metric("Detection %", f"{summary['pose_detection_percentage']}%")
        summary_cols[4].metric("Avg Visibility", summary["average_visibility_score"])

        st.video(annotated_video_path)

    st.divider()
    st.subheader("Strike Event Detection")
    st.caption("Detect generic strike-like events from saved pose landmark movement only.")

    sensitive_mode = st.checkbox(
        "Sensitive mode",
        help="Lowers the normalized speed threshold and duration requirements for testing.",
        key="event_detection_sensitive_mode",
    )
    default_speed_threshold = 0.45 if sensitive_mode else DEFAULT_SPEED_THRESHOLD
    default_min_duration = 0.08 if sensitive_mode else DEFAULT_MIN_EVENT_DURATION_SECONDS
    default_merge_gap = 0.18 if sensitive_mode else DEFAULT_MERGE_GAP_SECONDS
    default_min_visibility = 0.35 if sensitive_mode else DEFAULT_MIN_VISIBILITY_THRESHOLD
    default_smoothing = 2 if sensitive_mode else DEFAULT_SMOOTHING_WINDOW

    with st.expander("Advanced Detection Settings"):
        st.caption(
            "Speed values use MediaPipe's normalized x/y coordinates per second, not pixels."
        )
        speed_threshold = st.slider(
            "Speed threshold",
            min_value=0.1,
            max_value=3.0,
            value=float(default_speed_threshold),
            step=0.05,
            key="event_detection_speed_threshold",
        )
        min_event_duration_seconds = st.slider(
            "Minimum event duration (seconds)",
            min_value=0.02,
            max_value=1.0,
            value=float(default_min_duration),
            step=0.01,
            key="event_detection_min_duration",
        )
        merge_gap_seconds = st.slider(
            "Merge gap (seconds)",
            min_value=0.0,
            max_value=1.0,
            value=float(default_merge_gap),
            step=0.01,
            key="event_detection_merge_gap",
        )
        min_visibility_threshold = st.slider(
            "Minimum visibility threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(default_min_visibility),
            step=0.05,
            key="event_detection_visibility_threshold",
        )
        smoothing_window = st.slider(
            "Smoothing window (frames)",
            min_value=1,
            max_value=15,
            value=int(default_smoothing),
            step=1,
            key="event_detection_smoothing_window",
        )

    if st.button("Detect Strike Events"):
        pose_csv_path = build_processed_pose_csv_path(selected_video_path)
        strike_events_csv_path = build_processed_strike_events_csv_path(selected_video_path)

        if not pose_csv_path.exists():
            st.error("Pose CSV not found. Extract pose data before running strike detection.")
            return

        try:
            pose_df = pd.read_csv(pose_csv_path)
            strike_events_df = detect_strike_events(
                pose_df=pose_df,
                speed_threshold=float(speed_threshold),
                min_event_duration_seconds=float(min_event_duration_seconds),
                merge_gap_seconds=float(merge_gap_seconds),
                min_visibility_threshold=float(min_visibility_threshold),
                smoothing_window=int(smoothing_window),
            )
            strike_events_df["analysis_mode"] = analysis_mode
            strike_events_df.to_csv(strike_events_csv_path, index=False)
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Strike event detection failed: {exc}")
            return

        total_events = len(strike_events_df)
        punch_count = int((strike_events_df["event_category"] == "punch_candidate").sum())
        kick_count = int((strike_events_df["event_category"] == "kick_candidate").sum())

        st.caption(f"Detected events saved to `{strike_events_csv_path.as_posix()}`")

        summary_cols = st.columns(3)
        summary_cols[0].metric("Total Events", total_events)
        summary_cols[1].metric("Punch Candidates", punch_count)
        summary_cols[2].metric("Kick Candidates", kick_count)

        if strike_events_df.empty:
            st.info("No strike-like movement events were detected with the current thresholds.")
            return

        st.write("Detected strike events:")
        st.dataframe(strike_events_df, use_container_width=True)

        timeline_chart = (
            alt.Chart(strike_events_df)
            .mark_bar(size=18)
            .encode(
                x=alt.X("start_time:Q", title="Video Time (seconds)"),
                x2="end_time:Q",
                y=alt.Y("event_id:N", title="Event ID"),
                color=alt.Color("event_category:N", title="Category"),
                tooltip=[
                    "event_id",
                    "primary_limb",
                    "event_category",
                    "start_time",
                    "end_time",
                    "peak_time",
                    "confidence_score",
                    "notes",
                ],
            )
            .properties(height=max(120, total_events * 28), title="Detected Event Timeline")
        )
        st.altair_chart(timeline_chart, use_container_width=True)


def render_clip_review_and_labeling_section(
    selected_video_path: Path | None,
    analysis_mode: str,
) -> None:
    """Render clip export, review, and label saving independently."""
    st.header("Clip Review & Labeling")

    if selected_video_path is None:
        render_video_required_message()
        return

    st.caption(
        f"Selected video: `{selected_video_path.name}` | Analysis mode: `{analysis_mode}`"
    )
    model_path = build_baseline_model_path()
    st.checkbox(
        "Use model suggestions when available",
        value=True,
        key="use_model_suggestions",
    )
    if model_path.exists():
        st.info("Model suggestions use the saved fine strike model.")
    st.subheader("Event Clip Export")
    st.caption("Export short review clips for each detected strike event.")
    force_reexport_clips = st.checkbox(
        "Force Re-export Clips",
        help="Delete existing clips for this video and regenerate them.",
        key="clip_review_force_reexport",
    )

    if st.button("Export Event Clips"):
        strike_events_csv_path = build_processed_strike_events_csv_path(selected_video_path)
        clips_output_dir = build_processed_clips_dir(selected_video_path)

        if not strike_events_csv_path.exists():
            st.error("Strike events CSV not found. Detect strike events before exporting clips.")
            return

        try:
            if force_reexport_clips and clips_output_dir.exists():
                shutil.rmtree(clips_output_dir)

            strike_events_df = pd.read_csv(strike_events_csv_path)
            exported_clips_df = export_event_clips(
                video_path=str(selected_video_path),
                events_df=strike_events_df,
                output_dir=str(clips_output_dir),
                pre_seconds=1.0,
                post_seconds=1.0,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Clip export failed: {exc}")
            return

        st.session_state["exported_clips_df"] = exported_clips_df
        st.session_state["exported_clips_video_name"] = selected_video_path.name
        st.success(f"Exported {len(exported_clips_df)} event clips to `{clips_output_dir.as_posix()}`")
        st.dataframe(exported_clips_df, use_container_width=True)

        if not bool(exported_clips_df["ffmpeg_available"].all()):
            st.warning(
                "FFmpeg is not available to this Python/Streamlit process. Restart PowerShell or add FFmpeg bin folder to PATH."
            )

        failed_ffmpeg_rows = exported_clips_df[
            (exported_clips_df["ffmpeg_available"] == True)
            & (exported_clips_df["codec_source"] == "opencv_raw")
        ]
        if not failed_ffmpeg_rows.empty:
            st.warning("FFmpeg conversion failed for one or more clips.")

    exported_clips_df = st.session_state.get("exported_clips_df")
    exported_clips_video_name = st.session_state.get("exported_clips_video_name")

    if (
        isinstance(exported_clips_df, pd.DataFrame)
        and not exported_clips_df.empty
        and exported_clips_video_name == selected_video_path.name
    ):
        review_clips_df = exported_clips_df.copy()
        if bool(st.session_state.get("use_model_suggestions", True)) and model_path.exists():
            review_clips_df, suggestion_warnings = build_model_suggestions_for_clips(
                review_clips_df,
                selected_video_path,
            )
            st.session_state["exported_clips_df"] = review_clips_df
            if suggestion_warnings:
                with st.expander("Model Suggestion Warnings"):
                    for warning in suggestion_warnings:
                        st.warning(warning)
        with st.form("clip_review_form", clear_on_submit=False):
            render_clip_review_section(review_clips_df)
            submitted = st.form_submit_button("Save Labels", type="primary")

        if submitted:
            labels_csv_path = build_event_labels_csv_path()

            try:
                added_count, attempted_count = save_event_labels(
                    video_filename=selected_video_path.name,
                    analysis_mode=analysis_mode,
                    clips_df=review_clips_df,
                    labels_csv_path=str(labels_csv_path),
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Saving labels failed: {exc}")
                return

            st.success(
                f"Saved {added_count} new label rows out of {attempted_count} reviewed clips "
                f"to `{labels_csv_path.as_posix()}`"
            )
    else:
        st.info("Export event clips for the selected video to start review and labeling.")


st.set_page_config(page_title="StrikeLens", page_icon="🥊", layout="wide")


def main() -> None:
    """Render independent StrikeLens workflows with shared project state."""
    ensure_directories()

    st.title("StrikeLens")
    st.caption(
        "Move between video processing, labeling, dataset QA, and model training without restarting the workflow."
    )

    selected_section = render_sidebar_navigation()
    selected_video_path, analysis_mode = render_video_selector()
    render_project_state_panel()

    if selected_section == "Dashboard":
        render_dashboard_section()
    elif selected_section == "Video Processing":
        render_video_processing_section(selected_video_path)
    elif selected_section == "Event Detection":
        render_event_detection_section(selected_video_path, analysis_mode)
    elif selected_section == "Clip Review & Labeling":
        render_clip_review_and_labeling_section(selected_video_path, analysis_mode)
    elif selected_section == "Dataset & Label Review":
        render_dataset_dashboard()
    elif selected_section == "Fine Strike Model":
        render_baseline_model_training_section()
    elif selected_section == "Video Holdout Validation":
        render_video_holdout_validation_section()


if __name__ == "__main__":
    main()
