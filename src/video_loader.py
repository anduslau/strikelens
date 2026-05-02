from pathlib import Path
from typing import BinaryIO

import cv2

from src.utils import UPLOADS_DIR


def is_valid_video_file(filename: str) -> bool:
    """Allow only MP4 uploads for the current phase."""
    return Path(filename).suffix.lower() == ".mp4"


def save_uploaded_video(uploaded_file: BinaryIO) -> Path:
    """Persist the uploaded file inside data/uploads and return its path."""
    target_path = UPLOADS_DIR / Path(uploaded_file.name).name

    try:
        target_path.write_bytes(uploaded_file.getbuffer())
    except Exception as exc:
        raise ValueError("Unable to save the uploaded video.") from exc

    return target_path


def extract_video_metadata(video_path: Path) -> dict[str, float | int | str]:
    """Read core metadata from a video file using OpenCV."""
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError("The uploaded file could not be opened as a valid video.")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Avoid dividing by zero when OpenCV cannot detect FPS.
        duration_seconds = frame_count / fps if fps and fps > 0 else 0.0

        return {
            "filename": video_path.name,
            "duration_seconds": round(duration_seconds, 2),
            "fps": round(fps, 2) if fps and fps > 0 else 0.0,
            "frame_count": frame_count,
            "resolution": f"{width} x {height}",
        }
    finally:
        capture.release()


def format_metadata(metadata: dict[str, float | int | str]) -> dict[str, str]:
    """Prepare metadata values for a clean Streamlit display."""
    return {
        "Filename": str(metadata["filename"]),
        "Duration (seconds)": str(metadata["duration_seconds"]),
        "FPS": str(metadata["fps"]),
        "Frame Count": str(metadata["frame_count"]),
        "Resolution": str(metadata["resolution"]),
    }
