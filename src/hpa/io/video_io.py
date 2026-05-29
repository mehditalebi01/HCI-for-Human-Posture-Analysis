"""Video input utilities for the sports posture analysis pipeline."""

from pathlib import Path

import cv2

from hpa.utils.paths import ensure_dir


def get_video_metadata(video_path):
    """Read basic metadata from a video file.

    Returns a dictionary with total frame count, FPS, width, height, and duration.
    """
    video_file = Path(video_path)

    if not video_file.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_file}")

    video = cv2.VideoCapture(str(video_file))
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0

    video.release()

    return {
        "video_path": str(video_file),
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
    }


def extract_frames(video_path, output_dir, step):
    """Save one frame every `step` frames from a video.

    Returns a result dictionary that scripts can print or tests can inspect.
    """
    video_file = Path(video_path)
    output_folder = ensure_dir(output_dir)

    if step <= 0:
        raise ValueError("Frame step must be a positive number.")

    metadata = get_video_metadata(video_file)

    video = cv2.VideoCapture(str(video_file))
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")

    saved_frames = 0
    frame_number = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        if frame_number % step == 0:
            frame_name = output_folder / f"frame_{frame_number:06d}.jpg"
            cv2.imwrite(str(frame_name), frame)
            saved_frames += 1

        frame_number += 1

    video.release()

    metadata["saved_frames"] = saved_frames
    metadata["output_dir"] = str(output_folder)
    return metadata
