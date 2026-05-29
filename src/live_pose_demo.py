import argparse
import time
from pathlib import Path

import cv2
import numpy as np


import os
from pathlib import Path

venv = Path(__file__).resolve().parent.parent / ".venv"

os.add_dll_directory(str(venv / "Lib/site-packages/nvidia/cudnn/bin"))
os.add_dll_directory(str(venv / "Lib/site-packages/nvidia/cublas/bin"))
import onnxruntime as ort
ort.preload_dlls(directory="")
print(ort.get_available_providers())

def resize_if_large(frame, max_width=960):
    """Resize large frames so pose estimation runs faster."""
    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = max_width / width
    new_height = int(height * scale)
    return cv2.resize(frame, (max_width, new_height))


def draw_fps(frame, fps):
    """Draw the current frames-per-second value in the top-left corner."""
    text = f"FPS: {fps:.1f}"
    position = (20, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw a dark shadow first so the text stays readable on bright videos.
    cv2.putText(frame, text, position, font, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, position, font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)


def run_live_pose_demo(video_path):
    """Read a video, estimate human poses, and show the result live."""
    video_file = Path(video_path)

    if not video_file.exists():
        print(f"Error: video file does not exist: {video_file}")
        return False

    try:
        from rtmlib import Body, draw_skeleton
    except ImportError:
        print("Error: rtmlib is not installed. Run: pip install -r requirements.txt")
        return False

    # Open the input video with OpenCV.
    video = cv2.VideoCapture(str(video_file))
    if not video.isOpened():
        print(f"Error: could not open video file: {video_file}")
        return False

    # CPU or cuda 
    device = "cuda"
    backend = "onnxruntime"
    openpose_skeleton = False
    pose_model = Body(
        to_openpose=openpose_skeleton,
        mode="balanced",
        backend=backend,
        device=device,
    )

    window_name = "Live Pose Demo - press q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    processed_frames = 0

    try:
        while True:
            start_time = time.perf_counter()
            success, frame = video.read()

            if not success:
                if processed_frames == 0:
                    print("Error: failed to read the first frame from the video.")
                    return False

                print("Finished reading video.")
                break

            processed_frames += 1

            # Resize very large frames before pose estimation to improve speed.
            frame = resize_if_large(frame)

            # Run RTMPose on the current video frame.
            keypoints, scores = pose_model(frame)
            keypoints = np.asarray(keypoints)
            scores = np.asarray(scores)

            # Draw detected body skeletons and keypoints on the frame.
            if keypoints.size > 0 and scores.size > 0:
                # Keep the shape consistent: people x keypoints x coordinates.
                if keypoints.ndim == 2:
                    keypoints = keypoints[None, :, :]
                    scores = scores[None, :]

                annotated_frame = draw_skeleton(
                    frame,
                    keypoints,
                    scores,
                    kpt_thr=0.5,
                    openpose_skeleton=openpose_skeleton,
                )
            else:
                annotated_frame = frame.copy()

            elapsed_time = time.perf_counter() - start_time
            fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
            draw_fps(annotated_frame, fps)

            cv2.imshow(window_name, annotated_frame)

            # Press q while the video window is active to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

    finally:
        video.release()
        cv2.destroyAllWindows()

    print(f"Processed frames: {processed_frames}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show live 2D human pose estimation on a sports video."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_live_pose_demo(args.video)

    if not success:
        raise SystemExit(1)
