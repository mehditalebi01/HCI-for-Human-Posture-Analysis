"""Pose smoothing and quality checks.

This module prepares raw 2D keypoints for later biomechanics steps. The current
implementation uses a simple moving average, which is easy to understand and a
good baseline before adding more advanced filters.
"""

import pandas as pd


REQUIRED_KEYPOINT_COLUMNS = {"frame_name", "person_id", "keypoint_id", "x", "y", "confidence"}


def _validate_keypoint_dataframe(df):
    """Check that the input data has the columns needed for smoothing."""
    missing_columns = REQUIRED_KEYPOINT_COLUMNS - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Keypoint CSV is missing required columns: {missing}")


def _add_sort_columns(df):
    """Add helper columns so rows are smoothed in frame order.

    Batch pose CSV files usually have `frame_name`. Live CSV files can also have
    `frame_index`. This function supports both.
    """
    sorted_df = df.copy()

    if "frame_index" in sorted_df.columns:
        sorted_df["_sort_frame"] = pd.to_numeric(sorted_df["frame_index"], errors="coerce")
    else:
        # frame_000030.jpg becomes 30. If parsing fails, pandas will keep NaN.
        extracted = sorted_df["frame_name"].astype(str).str.extract(r"(\d+)", expand=False)
        sorted_df["_sort_frame"] = pd.to_numeric(extracted, errors="coerce")

    # Keep original row order as a final fallback for unusual frame names.
    sorted_df["_original_order"] = range(len(sorted_df))
    sorted_df["_sort_frame"] = sorted_df["_sort_frame"].fillna(sorted_df["_original_order"])
    return sorted_df


def moving_average_smooth(df, window_size=5):
    """Smooth x and y coordinates per person and keypoint.

    Confidence values are not changed. A centered rolling window is used so each
    point is averaged with nearby frames from the same person/keypoint series.
    """
    _validate_keypoint_dataframe(df)

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    sorted_df = _add_sort_columns(df)
    sorted_df = sorted_df.sort_values(
        ["person_id", "keypoint_id", "_sort_frame", "_original_order"]
    )

    group_columns = ["person_id", "keypoint_id"]

    sorted_df["x"] = sorted_df.groupby(group_columns)["x"].transform(
        lambda values: values.rolling(
            window=window_size,
            min_periods=1,
            center=True,
        ).mean()
    )
    sorted_df["y"] = sorted_df.groupby(group_columns)["y"].transform(
        lambda values: values.rolling(
            window=window_size,
            min_periods=1,
            center=True,
        ).mean()
    )

    # Restore the original row order and remove helper columns before saving.
    sorted_df = sorted_df.sort_values("_original_order")
    return sorted_df.drop(columns=["_sort_frame", "_original_order"])


def compute_pose_quality(df, confidence_threshold=0.3):
    """Calculate simple quality metrics for a keypoint DataFrame."""
    _validate_keypoint_dataframe(df)

    if confidence_threshold < 0:
        raise ValueError("confidence_threshold must be zero or greater.")

    total_keypoints = len(df)
    average_confidence = float(df["confidence"].mean()) if total_keypoints > 0 else 0.0
    low_confidence_count = int((df["confidence"] < confidence_threshold).sum())
    low_confidence_ratio = (
        low_confidence_count / total_keypoints if total_keypoints > 0 else 0.0
    )
    frames_processed = int(df["frame_name"].nunique()) if total_keypoints > 0 else 0

    return {
        "total_keypoints": total_keypoints,
        "average_confidence": average_confidence,
        "low_confidence_count": low_confidence_count,
        "low_confidence_ratio": low_confidence_ratio,
        "frames_processed": frames_processed,
    }
