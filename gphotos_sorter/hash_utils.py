from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from PIL import Image
import imagehash

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def compute_hash(path: Path) -> Optional[str]:
    try:
        if is_image(path):
            with Image.open(path) as image:
                return f"phash:{imagehash.phash(image)}"
        return f"sha256:{sha256_file(path)}"
    except Exception:
        return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_image_resolution(path: Path) -> tuple[Optional[int], Optional[int]]:
    """Get image width and height, returns (width, height) or (None, None) if unable to read."""
    try:
        if is_image(path):
            with Image.open(path) as image:
                return image.size  # Returns (width, height)
        return None, None
    except Exception:
        return None, None
