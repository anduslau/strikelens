import shutil
import subprocess
from pathlib import Path

import cv2
import pandas as pd


FALLBACK_FFMPEG_PATH = Path(
    r"C:\Users\Andus\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
)


def _sanitize_filename_component(value: str) -> str:
    """Make text safe for use inside exported clip filenames."""
    sanitized = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in value
    )
    return sanitized.strip("_") or "unknown"


def _resolve_ffmpeg_path() -> str | None:
    """Resolve ffmpeg from PATH first, then fall back to the known WinGet link."""
    resolved_path = shutil.which("ffmpeg")
    if resolved_path:
        return resolved_path
    if FALLBACK_FFMPEG_PATH.exists():
        return str(FALLBACK_FFMPEG_PATH)
    return None


def _convert_clip_with_ffmpeg(
    ffmpeg_path: str,
    raw_clip_path: Path,
    final_clip_path: Path,
) -> subprocess.CompletedProcess[str]:
    """Convert a raw MP4 clip into a browser-compatible H.264 MP4."""
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(raw_clip_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(final_clip_path),
    ]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )


def _validate_clip_file(clip_path: Path) -> tuple[str, int, float]:
    """Validate that a clip exists, has content, and can be read by OpenCV."""
    if not clip_path.exists():
        return "missing_file", 0, 0.0

    file_size = clip_path.stat().st_size
    if file_size <= 0:
        return "empty_file", 0, 0.0

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        return "opencv_open_failed", file_size, 0.0

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps and fps > 0 else 0.0

        if frame_count <= 0:
            return "no_frames", file_size, 0.0
        if duration <= 0:
            return "zero_duration", file_size, 0.0

        return "valid", file_size, round(duration, 4)
    finally:
        capture.release()


def export_event_clips(
    video_path: str,
    events_df: pd.DataFrame,
    output_dir: str,
    pre_seconds: float = 1.0,
    post_seconds: float = 1.0,
) -> pd.DataFrame:
    """Export short review clips around each detected event and return clip metadata."""
    input_path = Path(video_path)
    clips_output_dir = Path(output_dir)
    raw_output_dir = clips_output_dir / "raw"

    if not input_path.exists():
        raise ValueError("The source video does not exist.")
    if events_df.empty:
        raise ValueError("No strike events are available for clip export.")
    if pre_seconds < 0 or post_seconds < 0:
        raise ValueError("Pre-roll and post-roll seconds must be non-negative.")

    required_columns = {
        "event_id",
        "event_category",
        "primary_limb",
        "confidence_score",
        "start_time",
        "end_time",
    }
    missing_columns = sorted(required_columns - set(events_df.columns))
    if missing_columns:
        raise ValueError(
            "Events data is missing required columns: " + ", ".join(missing_columns)
        )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("The source video could not be opened for clip export.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        capture.release()
        raise ValueError("Unable to determine source video FPS for clip export.")
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise ValueError("Unable to determine source video resolution for clip export.")

    duration_seconds = frame_count / fps if frame_count > 0 else 0.0
    clips_output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = _resolve_ffmpeg_path()
    ffmpeg_available = ffmpeg_path is not None

    clip_rows: list[dict[str, str | int | float]] = []

    try:
        for event in events_df.to_dict(orient="records"):
            start_time = max(0.0, float(event["start_time"]) - pre_seconds)
            end_time = min(duration_seconds, float(event["end_time"]) + post_seconds)

            if end_time <= start_time:
                continue

            start_frame = max(0, int(start_time * fps))
            end_frame = min(frame_count - 1, int(end_time * fps))
            timestamp_label = f"{start_time:.2f}".replace(".", "p")
            filename = (
                f"event_{int(event['event_id']):03d}_"
                f"{_sanitize_filename_component(str(event['event_category']))}_"
                f"{_sanitize_filename_component(str(event['primary_limb']))}_"
                f"t{timestamp_label}.mp4"
            )
            raw_clip_path = raw_output_dir / filename
            final_clip_path = clips_output_dir / filename

            writer = cv2.VideoWriter(
                str(raw_clip_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame_width, frame_height),
            )
            if not writer.isOpened():
                raise ValueError(f"Unable to create clip file: {raw_clip_path.name}")

            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current_frame = start_frame

                while current_frame <= end_frame:
                    success, frame = capture.read()
                    if not success:
                        break
                    writer.write(frame)
                    current_frame += 1
            finally:
                writer.release()

            codec_source = "opencv_raw"
            selected_clip_path = raw_clip_path
            ffmpeg_command = ""
            ffmpeg_return_code = -1
            ffmpeg_stdout = ""
            ffmpeg_stderr = ""

            if ffmpeg_available:
                conversion_result = _convert_clip_with_ffmpeg(
                    ffmpeg_path=ffmpeg_path,
                    raw_clip_path=raw_clip_path,
                    final_clip_path=final_clip_path,
                )
                ffmpeg_command = " ".join(
                    [
                        ffmpeg_path,
                        "-y",
                        "-i",
                        str(raw_clip_path),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        "-an",
                        str(final_clip_path),
                    ]
                )
                ffmpeg_return_code = int(conversion_result.returncode)
                ffmpeg_stdout = conversion_result.stdout.strip()
                ffmpeg_stderr = conversion_result.stderr.strip()

                if conversion_result.returncode == 0:
                    codec_source = "ffmpeg_h264"
                    selected_clip_path = final_clip_path
                else:
                    codec_source = "opencv_raw"

            validation_status, file_size, clip_duration = _validate_clip_file(selected_clip_path)

            clip_rows.append(
                {
                    "event_id": int(event["event_id"]),
                    "clip_path": str(selected_clip_path),
                    "event_category": str(event["event_category"]),
                    "primary_limb": str(event["primary_limb"]),
                    "analysis_mode": str(
                        event.get("analysis_mode", "single_athlete_training")
                    ),
                    "confidence_score": float(event["confidence_score"]),
                    "start_time": round(float(event["start_time"]), 4),
                    "end_time": round(float(event["end_time"]), 4),
                    "file_size": int(file_size),
                    "duration": float(clip_duration),
                    "codec_source": codec_source,
                    "validation_status": validation_status,
                    "ffmpeg_available": bool(ffmpeg_available),
                    "ffmpeg_path": ffmpeg_path or "",
                    "ffmpeg_command": ffmpeg_command,
                    "ffmpeg_return_code": int(ffmpeg_return_code),
                    "ffmpeg_stdout": ffmpeg_stdout,
                    "ffmpeg_stderr": ffmpeg_stderr,
                }
            )

        if not clip_rows:
            raise ValueError("No event clips could be exported from the detected events.")

        return pd.DataFrame(
            clip_rows,
            columns=[
                "event_id",
                "clip_path",
                "event_category",
                "primary_limb",
                "analysis_mode",
                "confidence_score",
                "start_time",
                "end_time",
                "file_size",
                "duration",
                "codec_source",
                "validation_status",
                "ffmpeg_available",
                "ffmpeg_path",
                "ffmpeg_command",
                "ffmpeg_return_code",
                "ffmpeg_stdout",
                "ffmpeg_stderr",
            ],
        )
    finally:
        capture.release()
