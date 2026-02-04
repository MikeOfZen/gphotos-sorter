from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFile
import imagehash

# Allow loading truncated/incomplete images instead of raising exceptions
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def compute_hash(path: Path) -> Optional[str]:
    """Compute perceptual hash for images, SHA256 for videos.
    
    Returns None if unable to compute hash (corrupt file, unsupported format, etc).
    """
    try:
        if is_image(path):
            # Set a reasonable decompression bomb limit (default is 89MB, we'll use 256MB)
            Image.MAX_IMAGE_PIXELS = 256 * 1024 * 1024 // 4  # 256MB / 4 bytes per pixel
            with Image.open(path) as image:
                # Force load to catch issues early
                image.load()
                return f"phash:{imagehash.phash(image)}"
        return f"sha256:{sha256_file(path)}"
    except Exception:
        # Any error: corrupt file, unsupported format, too large, etc
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


def verify_hash_collision(path1: Path, path2: Path) -> bool:
    """Verify if two files with the same perceptual hash are actually the same image.
    
    For pHash collisions, we compare SHA256 of the files.
    Returns True if files are genuinely the same (or close enough), False if collision.
    """
    try:
        # If file sizes differ significantly, likely different images
        size1 = path1.stat().st_size
        size2 = path2.stat().st_size
        size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0
        
        # If sizes are very different (less than 50% similar), likely different files
        if size_ratio < 0.5:
            return False
        
        # Compare SHA256 for definitive answer
        hash1 = sha256_file(path1)
        hash2 = sha256_file(path2)
        return hash1 == hash2
    except Exception:
        # If we can't verify, assume they're the same to avoid duplicates
        return True
