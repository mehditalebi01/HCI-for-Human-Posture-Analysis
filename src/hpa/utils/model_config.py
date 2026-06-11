"""Helpers for reading local model presets from YAML config files."""

from pathlib import Path

import yaml


def _to_tuple(value, default):
    """Convert a YAML list such as [192, 256] into a tuple."""
    if value is None:
        return default

    return tuple(value)


def load_model_preset(mode="balanced", config_path="configs/models.yaml"):
    """Load detector and pose model settings for one RTMLib mode.

    The project stores one detector and one pose model per mode so experiments
    are explicit and reproducible.
    """
    path = Path(config_path)

    if not path.exists():
        return {
            "det_model": None,
            "pose_model": None,
            "det_input_size": (640, 640),
            "pose_input_size": (192, 256),
        }

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    presets = config.get("mode_presets", {})
    preset = presets.get(mode)

    # Backward compatibility for the earlier single-model config format.
    if preset is None:
        preset = {
            "detection": config.get("detection", {}),
            "pose": config.get("pose", {}),
        }

    detection = preset.get("detection", {})
    pose = preset.get("pose", {})

    return {
        "det_model": detection.get("path"),
        "pose_model": pose.get("path"),
        "det_input_size": _to_tuple(detection.get("input_size"), (640, 640)),
        "pose_input_size": _to_tuple(pose.get("input_size"), (192, 256)),
    }
