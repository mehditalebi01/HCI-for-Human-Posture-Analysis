import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def find_frame_files(input_folder):
    """Find image frames in the input folder."""
    image_extensions = {".jpg", ".jpeg", ".png"}
    return sorted(
        frame
        for frame in input_folder.iterdir()
        if frame.is_file() and frame.suffix.lower() in image_extensions
    )


def save_keypoints_to_csv(csv_path, rows):
    """Write all detected keypoints to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_name", "person_id", "keypoint_id", "x", "y", "confidence"])
        writer.writerows(rows)


def estimate_poses(input_dir, output_csv, output_frames):
    """Run 2D body pose estimation on extracted image frames."""
    input_folder = Path(input_dir)
    csv_path = Path(output_csv)
    output_frame_folder = Path(output_frames)

    if not input_folder.exists():
        print(f"Error: input folder does not exist: {input_folder}")
        return False

    frame_files = find_frame_files(input_folder)
    if not frame_files:
        print(f"Error: no .jpg or .png frames found in: {input_folder}")
        return False

    output_frame_folder.mkdir(parents=True, exist_ok=True)

    try:
        from rtmlib import Body, draw_skeleton
    except ImportError:
        print("Error: rtmlib is not installed. Run: pip install -r requirements.txt")
        return False

    # RTMLib downloads/loads the selected model the first time it is used.
    # CPU is easiest for beginners. Change device to "cuda" if you have a GPU setup.
    device = "cuda"
    backend = "onnxruntime"
    openpose_skeleton = False
    pose_model = Body(
        to_openpose=openpose_skeleton,
        mode="balanced",
        backend=backend,
        device=device,
    )

    csv_rows = []

    for frame_path in tqdm(frame_files, desc="Estimating poses"):
        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"Warning: could not read image, skipping: {frame_path}")
            continue

        # keypoints shape: number of people x number of keypoints x 2
        # scores shape: number of people x number of keypoints
        keypoints, scores = pose_model(image)
        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)

        if keypoints.size > 0 and scores.size > 0:
            # Some pose libraries return one person as 2D data.
            # This keeps the CSV loop consistent: people x keypoints x coordinates.
            if keypoints.ndim == 2:
                keypoints = keypoints[None, :, :]
                scores = scores[None, :]

            for person_id, person_keypoints in enumerate(keypoints):
                for keypoint_id, point in enumerate(person_keypoints):
                    x, y = point
                    confidence = scores[person_id][keypoint_id]
                    csv_rows.append(
                        [
                            frame_path.name,
                            person_id,
                            keypoint_id,
                            float(x),
                            float(y),
                            float(confidence),
                        ]
                    )

        # Draw the detected skeleton on a copy of the original image.
        annotated_image = image.copy()
        annotated_image = draw_skeleton(
            annotated_image,
            keypoints,
            scores,
            kpt_thr=0.5,
            openpose_skeleton=openpose_skeleton,
        )

        output_image_path = output_frame_folder / frame_path.name
        cv2.imwrite(str(output_image_path), annotated_image)

    save_keypoints_to_csv(csv_path, csv_rows)

    print(f"Processed frames: {len(frame_files)}")
    print(f"Saved keypoints CSV: {csv_path}")
    print(f"Saved annotated frames: {output_frame_folder}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 2D human body pose estimation on extracted video frames."
    )
    parser.add_argument("--input", required=True, help="Folder containing extracted frames.")
    parser.add_argument("--output-csv", required=True, help="Path to save keypoints CSV.")
    parser.add_argument(
        "--output-frames",
        required=True,
        help="Folder to save frames with skeletons drawn on them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = estimate_poses(args.input, args.output_csv, args.output_frames)

    if not success:
        raise SystemExit(1)
