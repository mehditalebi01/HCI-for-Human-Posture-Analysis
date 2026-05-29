# Stage 02: 2D Pose Estimation

Stage 2 estimates 2D human body keypoints on extracted frames using RTMLib.

## Current Implementation

- Reusable estimator: `src/hpa/pose/pose_estimator.py`
- Visualization helpers: `src/hpa/pose/visualization.py`
- Batch CLI entry point: `src/scripts/run_pose_on_frames.py`
- Live demo CLI entry point: `src/scripts/live_pose_demo.py`
- Compatibility wrappers: `src/pose_estimation.py` and `src/live_pose_demo.py`

## Model Management

Model paths are defined in `configs/models.yaml`. Download local model files with:

```bash
python src/scripts/download_models.py
```

If local model files are missing, the estimator warns the user and allows RTMLib default behavior as a fallback.

## Batch Pose Command

```bash
python src/scripts/run_pose_on_frames.py --input data/interim/frames --output-csv data/processed/keypoints/keypoints.csv --output-frames data/outputs/annotated_frames
```

## CSV Output

The keypoint CSV uses this schema:

```text
frame_name, person_id, keypoint_id, x, y, confidence
```

## Live Demo Command

```bash
python src/scripts/live_pose_demo.py --video data/raw/videos/test.mp4
```
