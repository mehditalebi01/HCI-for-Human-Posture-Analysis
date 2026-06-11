import argparse
import sys
import time
from pathlib import Path

import cv2

# Allow running this file directly before `pip install -e .`.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hpa.pose.pose_estimator import RTMPoseEstimator
from hpa.pose.visualization import draw_fps, draw_pose
from hpa.io.keypoint_io import keypoints_to_rows, save_keypoints_csv
from hpa.utils.model_config import load_model_preset
from hpa.utils.paths import ensure_dir


def resize_if_large(frame, max_width):
    """Resize large frames so live inference runs faster."""
    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = max_width / width
    new_height = int(height * scale)
    return cv2.resize(frame, (max_width, new_height))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show live 2D human pose estimation on a video."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--device", default="cpu", help="RTMLib device: cpu or cuda.")
    parser.add_argument("--backend", default="onnxruntime", help="RTMLib backend.")
    parser.add_argument(
        "--mode",
        default="balanced",
        choices=["performance", "balanced", "lightweight"],
        help="RTMLib model preset.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=960,
        help="Resize frames wider than this value before inference.",
    )
    parser.add_argument("--det-model", default=None, help="Optional local detector model.")
    parser.add_argument("--pose-model", default=None, help="Optional local pose model.")
    parser.add_argument(
        "--output-csv",
        default="data/processed/keypoints/live_keypoints.csv",
        help="Path for live keypoint CSV output.",
    )
    parser.add_argument(
        "--output-video",
        default="data/outputs/videos/live_pose_demo.mp4",
        help="Path for the annotated output video.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Show the live demo without saving CSV or video outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_file = Path(args.video)

    if not video_file.exists():
        print(f"Error: video file does not exist: {video_file}")
        return 1

    video = cv2.VideoCapture(str(video_file))
    if not video.isOpened():
        print(f"Error: could not open video file: {video_file}")
        return 1

    model_preset = load_model_preset(args.mode)
    det_model = args.det_model or model_preset["det_model"]
    pose_model = args.pose_model or model_preset["pose_model"]

    estimator = RTMPoseEstimator(
        device=args.device,
        backend=args.backend,
        mode=args.mode,
        det_model_path=det_model,
        pose_model_path=pose_model,
        det_input_size=model_preset["det_input_size"],
        pose_input_size=model_preset["pose_input_size"],
    )

    window_name = "Live Pose Demo - press q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    processed_frames = 0
    csv_rows = []
    video_writer = None
    input_fps = video.get(cv2.CAP_PROP_FPS)
    output_fps = input_fps if input_fps > 0 else 30.0

    if not args.no_save:
        ensure_dir(Path(args.output_csv).parent)
        ensure_dir(Path(args.output_video).parent)

    try:
        while True:
            start_time = time.perf_counter()
            success, frame = video.read()

            if not success:
                if processed_frames == 0:
                    print("Error: failed to read the first frame from the video.")
                    return 1

                print("Finished reading video.")
                break

            processed_frames += 1
            frame = resize_if_large(frame, args.max_width)
            time_sec = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            keypoints, scores = estimator.estimate_image(frame)
            csv_rows.extend(
                keypoints_to_rows(
                    frame_name=f"frame_{processed_frames:06d}",
                    keypoints=keypoints,
                    scores=scores,
                    frame_index=processed_frames,
                    time_sec=time_sec,
                    include_timing=True,
                )
            )
            annotated_frame = draw_pose(
                frame,
                keypoints,
                scores,
                openpose_skeleton=estimator.openpose_skeleton,
            )

            elapsed_time = time.perf_counter() - start_time
            fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
            draw_fps(annotated_frame, fps)

            if not args.no_save:
                if video_writer is None:
                    height, width = annotated_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(args.output_video),
                        fourcc,
                        output_fps,
                        (width, height),
                    )

                video_writer.write(annotated_frame)

            cv2.imshow(window_name, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

    finally:
        video.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    if not args.no_save:
        save_keypoints_csv(args.output_csv, csv_rows, include_timing=True)
        print(f"Saved live keypoints CSV: {args.output_csv}")
        print(f"Saved annotated video: {args.output_video}")

    print(f"Processed frames: {processed_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
