"""Helpers for saving pose keypoints."""

import csv
from pathlib import Path

from hpa.utils.paths import ensure_dir


def keypoints_to_rows(
    frame_name,
    keypoints,
    scores,
    frame_index=None,
    time_sec=None,
    include_timing=False,
):
    """Convert pose model outputs into CSV rows.

    `frame_index` and `time_sec` are optional so the same helper can be used for
    static image folders and live/video processing.
    """
    rows = []

    if keypoints.size == 0 or scores.size == 0:
        return rows

    for person_id, person_keypoints in enumerate(keypoints):
        for keypoint_id, point in enumerate(person_keypoints):
            x, y = point
            confidence = scores[person_id][keypoint_id]
            if include_timing:
                rows.append(
                    [
                        frame_name,
                        frame_index,
                        time_sec,
                        person_id,
                        keypoint_id,
                        float(x),
                        float(y),
                        float(confidence),
                    ]
                )
            else:
                rows.append(
                    [
                        frame_name,
                        person_id,
                        keypoint_id,
                        float(x),
                        float(y),
                        float(confidence),
                    ]
                )

    return rows


def save_keypoints_csv(csv_path, rows, include_timing=False):
    """Save keypoint rows using the pipeline CSV schema."""
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if include_timing:
            writer.writerow(
                [
                    "frame_name",
                    "frame_index",
                    "time_sec",
                    "person_id",
                    "keypoint_id",
                    "x",
                    "y",
                    "confidence",
                ]
            )
        else:
            writer.writerow(["frame_name", "person_id", "keypoint_id", "x", "y", "confidence"])
        writer.writerows(rows)
