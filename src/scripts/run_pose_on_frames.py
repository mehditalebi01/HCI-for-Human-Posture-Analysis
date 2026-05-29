import argparse
import sys
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

# Allow running this file directly before `pip install -e .`.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hpa.pose.pose_estimator import RTMPoseEstimator
from hpa.pose.visualization import draw_pose
from hpa.io.keypoint_io import keypoints_to_rows, save_keypoints_csv
from hpa.utils.paths import ensure_dir, list_images


def load_default_model_paths(config_path="configs/models.yaml"):
    """Read local model paths from the model config file."""
    path = Path(config_path)

    if not path.exists():
        return None, None

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    det_path = config.get("detection", {}).get("path")
    pose_path = config.get("pose", {}).get("path")
    return det_path, pose_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 2D human pose estimation on extracted frames."
    )
    parser.add_argument(
        "--input",
        default="data/interim/frames",
        help="Folder containing frames.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/keypoints/keypoints.csv",
        help="Path for keypoints CSV.",
    )
    parser.add_argument(
        "--output-frames",
        default="data/outputs/annotated_frames",
        help="Folder for annotated skeleton frames.",
    )
    parser.add_argument("--device", default="cpu", help="RTMLib device: cpu or cuda.")
    parser.add_argument("--backend", default="onnxruntime", help="RTMLib backend.")
    parser.add_argument(
        "--mode",
        default="balanced",
        choices=["performance", "balanced", "lightweight"],
        help="RTMLib model preset.",
    )
    parser.add_argument("--det-model", default=None, help="Optional local detector model.")
    parser.add_argument("--pose-model", default=None, help="Optional local pose model.")
    return parser.parse_args()


def main():
    args = parse_args()
    frame_paths = list_images(args.input)

    if not frame_paths:
        print(f"Error: no .jpg or .png frames found in: {args.input}")
        return 1

    default_det_model, default_pose_model = load_default_model_paths()
    det_model = args.det_model or default_det_model
    pose_model = args.pose_model or default_pose_model

    estimator = RTMPoseEstimator(
        device=args.device,
        backend=args.backend,
        mode=args.mode,
        det_model_path=det_model,
        pose_model_path=pose_model,
    )

    output_folder = ensure_dir(args.output_frames)
    csv_rows = []

    for frame_path in tqdm(frame_paths, desc="Estimating poses"):
        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"Warning: could not read image, skipping: {frame_path}")
            continue

        keypoints, scores = estimator.estimate_image(image)
        csv_rows.extend(keypoints_to_rows(frame_path.name, keypoints, scores))

        annotated_image = draw_pose(
            image,
            keypoints,
            scores,
            openpose_skeleton=estimator.openpose_skeleton,
        )
        cv2.imwrite(str(output_folder / frame_path.name), annotated_image)

    save_keypoints_csv(args.output_csv, csv_rows)

    print(f"Processed frames: {len(frame_paths)}")
    print(f"Saved keypoints CSV: {args.output_csv}")
    print(f"Saved annotated frames: {output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
