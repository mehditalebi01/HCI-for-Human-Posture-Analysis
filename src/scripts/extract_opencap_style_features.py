import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Allow running this file directly before `pip install -e .`.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hpa.biomechanics.opencap_features import extract_opencap_style_features
from hpa.utils.paths import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract OpenCap-inspired 2D biomechanical surrogate features."
    )
    parser.add_argument(
        "--input",
        default="data/processed/smoothed_keypoints/smoothed_keypoints.csv",
        help="Path to smoothed 2D keypoints CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/biomechanics/opencap_style_features.csv",
        help="Path to save biomechanical feature CSV.",
    )
    parser.add_argument(
        "--plots-dir",
        default="data/outputs/reports/opencap_style_plots",
        help="Folder where diagnostic plots will be saved.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Mean pose confidence below this value is flagged as low quality.",
    )
    return parser.parse_args()


def _frame_axis(features_df):
    """Choose the best available x-axis for plots."""
    if "frame_index" in features_df.columns:
        return features_df["frame_index"]

    return range(len(features_df))


def _save_line_plot(features_df, columns, title, ylabel, output_path):
    """Save a simple line plot for one or more feature columns."""
    plt.figure(figsize=(12, 6))

    for person_id, person_df in features_df.groupby("person_id", sort=False):
        x_axis = _frame_axis(person_df)

        for column in columns:
            if column in person_df.columns:
                plt.plot(x_axis, person_df[column], label=f"person {person_id} {column}")

    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=(0, 0, 0.78, 1))
    plt.savefig(output_path)
    plt.close()


def save_feature_plots(features_df, plots_dir):
    """Create diagnostic plots for extracted biomechanical features."""
    plots_path = ensure_dir(plots_dir)

    _save_line_plot(
        features_df,
        ["left_knee_angle", "right_knee_angle"],
        "2D Knee Angles",
        "Angle (degrees)",
        plots_path / "knee_angles.png",
    )
    _save_line_plot(
        features_df,
        ["left_hip_angle", "right_hip_angle"],
        "2D Hip Angles",
        "Angle (degrees)",
        plots_path / "hip_angles.png",
    )
    _save_line_plot(
        features_df,
        ["trunk_lean_angle"],
        "2D Trunk Lean Angle",
        "Angle from vertical (degrees)",
        plots_path / "trunk_lean.png",
    )
    _save_line_plot(
        features_df,
        ["knee_angle_asymmetry", "hip_angle_asymmetry"],
        "Left/Right Angle Asymmetry",
        "Absolute difference (degrees)",
        plots_path / "asymmetry.png",
    )


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: input smoothed keypoints CSV does not exist: {input_path}")
        return 1

    try:
        keypoints_df = pd.read_csv(input_path)
        features_df = extract_opencap_style_features(
            keypoints_df,
            confidence_threshold=args.confidence_threshold,
        )
    except (ValueError, pd.errors.EmptyDataError) as error:
        print(f"Error: {error}")
        return 1

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    features_df.to_csv(output_path, index=False)
    save_feature_plots(features_df, args.plots_dir)

    print("Extracted OpenCap-inspired 2D surrogate features.")
    print(f"Input keypoints: {input_path}")
    print(f"Saved features: {output_path}")
    print(f"Saved plots: {args.plots_dir}")
    print(f"Rows: {len(features_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
