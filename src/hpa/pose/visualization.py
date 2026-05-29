"""Pose visualization helpers."""

import cv2
import numpy as np


def draw_pose(image, keypoints, scores=None, keypoint_threshold=0.5, openpose_skeleton=False):
    """Draw body keypoints and skeleton lines on an image."""
    output_image = image.copy()
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores) if scores is not None else None

    if keypoints.size == 0 or scores is None or scores.size == 0:
        return output_image

    if keypoints.ndim == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :]

    from rtmlib import draw_skeleton

    return draw_skeleton(
        output_image,
        keypoints,
        scores,
        openpose_skeleton=openpose_skeleton,
        kpt_thr=keypoint_threshold,
    )


def draw_fps(image, fps):
    """Draw frames-per-second text on an image."""
    text = f"FPS: {fps:.1f}"
    position = (20, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(image, text, position, font, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, position, font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return image
