from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import pandas as pd

from src.pose_extractor import calculate_average_visibility


LOW_VISIBILITY_THRESHOLD = 0.6


def _draw_status_text(frame, message: str, color: tuple[int, int, int], y_position: int) -> None:
    """Draw a readable diagnostic banner on a video frame."""
    cv2.putText(
        frame,
        message,
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        cv2.LINE_AA,
    )


def build_pose_summary(diagnostics_df: pd.DataFrame) -> dict[str, float | int]:
    """Summarize pose-detection coverage for display in the app."""
    total_frames = int(len(diagnostics_df))
    detected_frames = int(diagnostics_df["pose_detected"].sum())
    missing_frames = total_frames - detected_frames
    detection_percentage = (detected_frames / total_frames * 100) if total_frames else 0.0
    average_visibility = (
        float(diagnostics_df["average_visibility_score"].mean()) if total_frames else 0.0
    )

    return {
        "total_frames": total_frames,
        "frames_with_pose_detected": detected_frames,
        "frames_with_pose_missing": missing_frames,
        "pose_detection_percentage": round(detection_percentage, 2),
        "average_visibility_score": round(average_visibility, 4),
    }


def create_pose_overlay_video(
    input_video_path: str,
    output_video_path: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> str:
    """Generate a video with MediaPipe pose landmarks drawn over each frame."""
    return create_pose_overlay_video_with_diagnostics(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        progress_callback=progress_callback,
    )["output_video_path"]


def create_pose_overlay_video_with_diagnostics(
    input_video_path: str,
    output_video_path: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, str | pd.DataFrame | dict[str, float | int]]:
    """Generate an annotated video plus per-frame pose diagnostics."""
    input_path = Path(input_video_path)
    output_path = Path(output_video_path)

    if not input_path.exists():
        raise ValueError("The input video does not exist.")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened for overlay generation.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        capture.release()
        raise ValueError("Unable to determine video FPS for overlay generation.")
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise ValueError("Unable to determine video resolution for overlay generation.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        capture.release()
        raise ValueError("The annotated video file could not be created.")

    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    pose_connections = mp.solutions.pose.POSE_CONNECTIONS
    pose_detector = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    detected_landmark_frames = 0
    processed_frames = 0
    diagnostics_rows: list[dict[str, float | int | bool | str]] = []

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            processed_frames += 1
            if progress_callback is not None:
                progress_callback(processed_frames, total_frames)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb_frame)
            timestamp_seconds = round((processed_frames - 1) / fps, 4)
            pose_detected = bool(results.pose_landmarks)
            average_visibility_score = 0.0
            status = "pose_missing"

            if pose_detected:
                average_visibility_score = round(
                    calculate_average_visibility(results.pose_landmarks.landmark),
                    4,
                )
                drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=results.pose_landmarks,
                    connections=pose_connections,
                    landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                )
                detected_landmark_frames += 1
                status = "pose_detected"

                # Fast strikes can briefly lower landmark confidence because motion blur
                # and self-occlusion make the athlete harder to track cleanly.
                if average_visibility_score < LOW_VISIBILITY_THRESHOLD:
                    _draw_status_text(frame, "Low confidence", (0, 215, 255), 40)
                    status = "low_confidence"
            else:
                _draw_status_text(frame, "Pose lost", (0, 0, 255), 40)

            writer.write(frame)
            diagnostics_rows.append(
                {
                    "frame_number": processed_frames,
                    "timestamp_seconds": timestamp_seconds,
                    "pose_detected": pose_detected,
                    "average_visibility_score": average_visibility_score,
                    "status": status,
                }
            )

        if processed_frames == 0:
            raise ValueError("No frames were processed from the uploaded video.")

        if detected_landmark_frames == 0:
            raise ValueError("No pose landmarks were detected in the video.")

        diagnostics_df = pd.DataFrame(diagnostics_rows)
        return {
            "output_video_path": str(output_path),
            "diagnostics_df": diagnostics_df,
            "summary": build_pose_summary(diagnostics_df),
        }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Pose overlay generation failed: {exc}") from exc
    finally:
        pose_detector.close()
        writer.release()
        capture.release()
