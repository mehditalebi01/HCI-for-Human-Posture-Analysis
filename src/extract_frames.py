"""Backward-compatible wrapper for the Stage 1 frame extraction script."""

from scripts.extract_frames import main


if __name__ == "__main__":
    raise SystemExit(main())
