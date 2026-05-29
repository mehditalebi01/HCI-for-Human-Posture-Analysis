"""OpenCap-inspired 2D biomechanical surrogate features.

These features are inspired by biomechanical workflows, but they are computed
from monocular 2D keypoints. They are not equivalent to full OpenCap 3D
kinematics or dynamics.
"""

import pandas as pd

from hpa.biomechanics.angles import (
    absolute_difference,
    angle_between_three_points,
    line_angle_from_vertical,
)


# RTMPose body7 uses the common COCO-17 body keypoint order.
COCO17 = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


REQUIRED_COLUMNS = {"frame_name", "person_id", "keypoint_id", "x", "y", "confidence"}


def _validate_keypoints(df):
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Keypoint CSV is missing required columns: {missing}")


def _point(keypoints_by_id, keypoint_name):
    """Return a named point as (x, y), or None if it is missing."""
    keypoint_id = COCO17[keypoint_name]
    row = keypoints_by_id.get(keypoint_id)

    if row is None:
        return None

    return (float(row["x"]), float(row["y"]))


def _angle_or_none(point_a, point_b, point_c):
    if point_a is None or point_b is None or point_c is None:
        return None

    return angle_between_three_points(point_a, point_b, point_c)


def _trunk_lean_or_none(left_shoulder, right_shoulder, left_hip, right_hip):
    if None in (left_shoulder, right_shoulder, left_hip, right_hip):
        return None

    shoulder_midpoint = (
        (left_shoulder[0] + right_shoulder[0]) / 2.0,
        (left_shoulder[1] + right_shoulder[1]) / 2.0,
    )
    hip_midpoint = (
        (left_hip[0] + right_hip[0]) / 2.0,
        (left_hip[1] + right_hip[1]) / 2.0,
    )
    return line_angle_from_vertical(shoulder_midpoint, hip_midpoint)


def extract_opencap_style_features(df, confidence_threshold=0.3):
    """Compute OpenCap-inspired 2D biomechanical features per frame/person."""
    _validate_keypoints(df)

    rows = []
    group_columns = ["frame_name", "person_id"]

    for (frame_name, person_id), group in df.groupby(group_columns, sort=False):
        keypoints_by_id = {
            int(row["keypoint_id"]): row
            for _, row in group.iterrows()
        }

        left_shoulder = _point(keypoints_by_id, "left_shoulder")
        right_shoulder = _point(keypoints_by_id, "right_shoulder")
        left_hip = _point(keypoints_by_id, "left_hip")
        right_hip = _point(keypoints_by_id, "right_hip")
        left_knee = _point(keypoints_by_id, "left_knee")
        right_knee = _point(keypoints_by_id, "right_knee")
        left_ankle = _point(keypoints_by_id, "left_ankle")
        right_ankle = _point(keypoints_by_id, "right_ankle")

        left_knee_angle = _angle_or_none(left_hip, left_knee, left_ankle)
        right_knee_angle = _angle_or_none(right_hip, right_knee, right_ankle)
        left_hip_angle = _angle_or_none(left_shoulder, left_hip, left_knee)
        right_hip_angle = _angle_or_none(right_shoulder, right_hip, right_knee)
        trunk_lean_angle = _trunk_lean_or_none(
            left_shoulder,
            right_shoulder,
            left_hip,
            right_hip,
        )

        mean_pose_confidence = float(group["confidence"].mean())
        low_confidence_flag = bool(mean_pose_confidence < confidence_threshold)

        output_row = {
            "frame_name": frame_name,
            "person_id": person_id,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "left_hip_angle": left_hip_angle,
            "right_hip_angle": right_hip_angle,
            "trunk_lean_angle": trunk_lean_angle,
            "knee_angle_asymmetry": absolute_difference(
                left_knee_angle,
                right_knee_angle,
            ),
            "hip_angle_asymmetry": absolute_difference(
                left_hip_angle,
                right_hip_angle,
            ),
            "mean_pose_confidence": mean_pose_confidence,
            "low_confidence_flag": low_confidence_flag,
        }

        if "frame_index" in group.columns:
            output_row["frame_index"] = group["frame_index"].iloc[0]

        if "time_sec" in group.columns:
            output_row["time_sec"] = group["time_sec"].iloc[0]

        rows.append(output_row)

    features_df = pd.DataFrame(rows)

    preferred_columns = [
        "frame_name",
        "frame_index",
        "time_sec",
        "person_id",
        "left_knee_angle",
        "right_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
        "trunk_lean_angle",
        "knee_angle_asymmetry",
        "hip_angle_asymmetry",
        "mean_pose_confidence",
        "low_confidence_flag",
    ]
    existing_columns = [column for column in preferred_columns if column in features_df.columns]
    return features_df[existing_columns]
