from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import pandas as pd


POSE_LANDMARKS = {
    "nose": mp.solutions.pose.PoseLandmark.NOSE,
    "left_shoulder": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp.solutions.pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
    "left_heel": mp.solutions.pose.PoseLandmark.LEFT_HEEL,
    "right_heel": mp.solutions.pose.PoseLandmark.RIGHT_HEEL,
    "left_foot_index": mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
    "right_foot_index": mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX,
}


def _build_empty_landmark_row() -> dict[str, float | None]:
    """Create a row template with null values for each tracked landmark field."""
    row: dict[str, float | None] = {}

    for landmark_name in POSE_LANDMARKS:
        row[f"{landmark_name}_x"] = None
        row[f"{landmark_name}_y"] = None
        row[f"{landmark_name}_z"] = None
        row[f"{landmark_name}_visibility"] = None

    return row


def calculate_average_visibility(landmarks) -> float:
    """Average visibility across the tracked landmarks used by StrikeLens."""
    visibility_values = [
        landmarks[landmark_enum.value].visibility for landmark_enum in POSE_LANDMARKS.values()
    ]
    return sum(visibility_values) / len(visibility_values)


def extract_pose_from_video(
    video_path: str,
    output_csv_path: str,
    sample_rate: int = 1,
    analysis_mode: str = "single_athlete_training",
) -> pd.DataFrame:
    """Extract pose landmarks from a video and save them as a CSV."""
    return extract_pose_from_video_with_progress(
        video_path=video_path,
        output_csv_path=output_csv_path,
        sample_rate=sample_rate,
        analysis_mode=analysis_mode,
    )


def extract_pose_from_video_with_progress(
    video_path: str,
    output_csv_path: str,
    sample_rate: int = 1,
    analysis_mode: str = "single_athlete_training",
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Extract pose landmarks from sampled frames and optionally report progress."""
    if sample_rate < 1:
        raise ValueError("Sample rate must be 1 or greater.")

    input_path = Path(video_path)
    output_path = Path(output_csv_path)

    if not input_path.exists():
        raise ValueError("The video file does not exist.")

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened for pose extraction.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        capture.release()
        raise ValueError("Unable to determine video FPS for pose extraction.")

    rows: list[dict[str, float | int | None]] = []
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        frame_number = 0

        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_number += 1

            if progress_callback is not None:
                progress_callback(frame_number, total_frames)

            if (frame_number - 1) % sample_rate != 0:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            row: dict[str, float | int | None] = {
                "frame_number": frame_number,
                "timestamp_seconds": round((frame_number - 1) / fps, 4),
                "analysis_mode": analysis_mode,
                "pose_detected": bool(results.pose_landmarks),
                "average_visibility_score": 0.0,
            }
            row.update(_build_empty_landmark_row())

            if results.pose_landmarks:
                # Fast strikes can cause temporary pose loss because motion blur and occlusion
                # make individual landmarks harder for MediaPipe to track consistently.
                row["average_visibility_score"] = round(
                    calculate_average_visibility(results.pose_landmarks.landmark),
                    4,
                )
                for landmark_name, landmark_enum in POSE_LANDMARKS.items():
                    landmark = results.pose_landmarks.landmark[landmark_enum.value]
                    row[f"{landmark_name}_x"] = landmark.x
                    row[f"{landmark_name}_y"] = landmark.y
                    row[f"{landmark_name}_z"] = landmark.z
                    row[f"{landmark_name}_visibility"] = landmark.visibility

            rows.append(row)

        if not rows:
            raise ValueError("No frames were processed from the uploaded video.")

        dataframe = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        return dataframe
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Pose extraction failed: {exc}") from exc
    finally:
        pose.close()
        capture.release()
