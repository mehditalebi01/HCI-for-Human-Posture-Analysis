import argparse
import sys
from pathlib import Path

# Allow running this file directly before `pip install -e .`.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hpa.io.video_io import extract_frames


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a sports video.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--output",
        default="data/interim/frames",
        help="Folder where frames will be saved.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=30,
        help="Save one frame every N frames. Example: 30 saves every 30th frame.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        result = extract_frames(args.video, args.output, args.step)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    print(f"Video: {result['video_path']}")
    print(f"Total frames: {result['total_frames']}")
    print(f"FPS: {result['fps']:.2f}")
    print(f"Duration: {result['duration']:.2f} seconds")
    print(f"Saved frames: {result['saved_frames']}")
    print(f"Output folder: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
