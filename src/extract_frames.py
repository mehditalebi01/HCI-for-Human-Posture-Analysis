import argparse
from pathlib import Path

import cv2


def extract_frames(video_path, output_dir, step):
    """Save one frame every `step` frames from a video."""
    video_file = Path(video_path)
    output_folder = Path(output_dir)

    if not video_file.exists():
        print(f"Error: video file does not exist: {video_file}")
        return

    if step <= 0:
        print("Error: --step must be a positive number.")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    # Open the video file with OpenCV.
    video = cv2.VideoCapture(str(video_file))
    if not video.isOpened():
        print(f"Error: could not open video file: {video_file}")
        return

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    saved_frames = 0
    frame_number = 0

    while True:
        success, frame = video.read()

        # Stop when there are no more frames to read.
        if not success:
            break

        if frame_number % step == 0:
            frame_name = output_folder / f"frame_{frame_number:06d}.jpg"
            cv2.imwrite(str(frame_name), frame)
            saved_frames += 1

        frame_number += 1

    video.release()

    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Saved frames: {saved_frames}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from a sports video at a fixed frame interval."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--output", required=True, help="Folder where frames will be saved.")
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Save one frame every N frames. Example: 30 saves every 30th frame.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frames(args.video, args.output, args.step)
