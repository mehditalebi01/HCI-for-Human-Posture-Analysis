# Stage 01: Video Input and Frame Extraction

Stage 1 provides the video input layer for the posture analysis pipeline.

## Current Implementation

- Reusable logic: `src/hpa/io/video_io.py`
- CLI entry point: `src/scripts/extract_frames.py`
- Compatibility wrapper: `src/extract_frames.py`

## Main Functions

- `get_video_metadata(video_path)` reads total frames, FPS, resolution, and duration.
- `extract_frames(video_path, output_dir, step)` saves one frame every N frames.

## Standard Command

```bash
python src/scripts/extract_frames.py --video data/raw/videos/test.mp4 --output data/interim/frames --step 30
```

## Output

Frames are saved to `data/interim/frames/` using names like `frame_000000.jpg`.
