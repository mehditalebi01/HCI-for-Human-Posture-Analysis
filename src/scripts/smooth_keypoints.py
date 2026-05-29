import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running this file directly before `pip install -e .`.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hpa.smoothing.smoother import compute_pose_quality, moving_average_smooth
from hpa.utils.paths import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smooth 2D pose keypoints and create a quality report."
    )
    parser.add_argument(
        "--input",
        default="data/processed/keypoints/keypoints.csv",
        help="Path to the raw keypoints CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/smoothed_keypoints/smoothed_keypoints.csv",
        help="Path to save the smoothed keypoints CSV.",
    )
    parser.add_argument(
        "--quality-report",
        default="data/outputs/reports/pose_quality.csv",
        help="Path to save pose quality metrics.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Moving average window size used for x and y smoothing.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Confidence value below which a keypoint is counted as low quality.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: input keypoints CSV does not exist: {input_path}")
        return 1

    try:
        keypoints_df = pd.read_csv(input_path)
        smoothed_df = moving_average_smooth(keypoints_df, window_size=args.window_size)
        quality_metrics = compute_pose_quality(
            keypoints_df,
            confidence_threshold=args.confidence_threshold,
        )
    except (ValueError, pd.errors.EmptyDataError) as error:
        print(f"Error: {error}")
        return 1

    output_path = Path(args.output)
    quality_report_path = Path(args.quality_report)

    ensure_dir(output_path.parent)
    ensure_dir(quality_report_path.parent)

    smoothed_df.to_csv(output_path, index=False)
    pd.DataFrame([quality_metrics]).to_csv(quality_report_path, index=False)

    print(f"Loaded keypoints: {input_path}")
    print(f"Saved smoothed keypoints: {output_path}")
    print(f"Saved quality report: {quality_report_path}")
    print(f"Total keypoints: {quality_metrics['total_keypoints']}")
    print(f"Average confidence: {quality_metrics['average_confidence']:.4f}")
    print(f"Low confidence ratio: {quality_metrics['low_confidence_ratio']:.4f}")
    print(f"Frames processed: {quality_metrics['frames_processed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
