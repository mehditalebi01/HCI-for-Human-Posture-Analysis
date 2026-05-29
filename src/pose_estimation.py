"""Backward-compatible wrapper for the Stage 2 pose estimation script."""

from scripts.run_pose_on_frames import main


if __name__ == "__main__":
    raise SystemExit(main())
