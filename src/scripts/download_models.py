import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

# Allow running this file directly before `pip install -e .`.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hpa.utils.paths import ensure_dir


def download_file(url, destination):
    """Download a file from a URL to a local path."""
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved archive: {destination}")


def find_first_onnx(folder):
    """Find the first ONNX file extracted from an archive."""
    onnx_files = sorted(Path(folder).rglob("*.onnx"))

    if not onnx_files:
        raise FileNotFoundError("No .onnx file found inside the downloaded archive.")

    return onnx_files[0]


def prepare_model(model_key, model_config):
    """Download, extract, and copy one configured model."""
    target_path = Path(model_config["path"])
    source_url = model_config["source_url"]
    model_name = model_config.get("name", model_key)

    if target_path.exists():
        print(f"Skipping {model_name}: already exists at {target_path}")
        return

    ensure_dir(target_path.parent)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        archive_path = temp_dir / f"{model_name}.zip"
        extract_dir = temp_dir / "extracted"

        download_file(source_url, archive_path)

        print(f"Extracting: {archive_path}")
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_dir)

        source_onnx = find_first_onnx(extract_dir)
        shutil.copy2(source_onnx, target_path)
        print(f"Prepared {model_name}: {target_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download configured model archives and prepare local ONNX files."
    )
    parser.add_argument(
        "--config",
        default="configs/models.yaml",
        help="Path to the model configuration YAML file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: model config not found: {config_path}")
        return 1

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    for model_key, model_config in config.items():
        prepare_model(model_key, model_config)

    print("Model preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
