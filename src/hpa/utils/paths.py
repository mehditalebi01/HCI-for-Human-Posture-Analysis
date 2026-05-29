"""Path helpers used across scripts and pipeline modules."""

from pathlib import Path


def ensure_dir(path):
    """Create a directory if needed and return it as a Path object."""
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def list_images(folder):
    """Return sorted JPG and PNG image files from a folder."""
    image_folder = Path(folder)
    image_extensions = {".jpg", ".jpeg", ".png"}

    if not image_folder.exists():
        return []

    return sorted(
        image_path
        for image_path in image_folder.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in image_extensions
    )
