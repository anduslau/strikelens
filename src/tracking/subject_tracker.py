"""Placeholder subject-tracking architecture for future multi-athlete support.

The current MVP assumes one clearly visible athlete. These functions define the
future insertion points for selecting and maintaining a primary subject across
frames once multi-person or sparring support is implemented.
"""


def select_primary_subject():
    """Select the main athlete to analyze from available subject candidates.

    Future implementations may rank candidates using position in frame,
    bounding-box size, motion continuity, or user selection.
    """


def track_subject_across_frames():
    """Maintain the selected athlete identity across consecutive frames.

    Future implementations may use keypoint similarity, bounding-box overlap,
    temporal smoothing, or tracker IDs to avoid switching between people.
    """


def handle_subject_switching():
    """Handle loss of the primary subject or accidental switches between people.

    Future implementations may pause analysis, re-lock to the last known
    subject, or flag low-confidence tracking segments for review.
    """
