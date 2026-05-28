# Sports Video Analysis Pipeline

This is a simple beginner-friendly computer vision project for extracting frames from sports videos and running 2D human pose estimation.

## Project Structure

```text
data/
  videos/   # Put input videos here
  frames/   # Extracted frames will be saved here
  output/   # Save processed results here later
src/        # Python source code
scripts/    # Helper scripts
```

## Install Dependencies

Create a virtual environment if you want to keep dependencies separate:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Extract Frames From a Video

Place a video file at:

```text
data/videos/test.mp4
```

Run the frame extraction script:

```bash
python src/extract_frames.py --video data/videos/test.mp4 --output data/frames --step 30
```

This command reads `data/videos/test.mp4` and saves one frame every 30 frames into `data/frames`.

## Run 2D Pose Estimation

After extracting frames, run pose estimation with RTMLib:

```bash
python src/pose_estimation.py --input data/frames --output-csv data/output/keypoints.csv --output-frames data/output/pose_frames
```

This command reads image frames from `data/frames`, detects human body keypoints, saves the keypoint data to `data/output/keypoints.csv`, and saves annotated skeleton images to `data/output/pose_frames`.
