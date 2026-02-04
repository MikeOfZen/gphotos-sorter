"""Perceptual hash processor.

Computes perceptual hash (pHash) for images and file hash for videos.
Used for duplicate detection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import imagehash
from PIL import Image

if TYPE_CHECKING:
    from ..core.protocols import ProcessingContext


# Supported image extensions for perceptual hashing
IMAGE_EXTENSIONS = frozenset({
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic', '.heif', '.tiff', '.tif'
})


class PerceptualHashProcessor:
    """Computes perceptual hash for images.
    
    For images: Uses imagehash library to compute pHash
    For videos: Uses file content hash (xxhash or md5)
    
    Perceptual hashes are resilient to:
    - Resizing
    - Minor color adjustments
    - Compression differences
    - Cropping (to some extent)
    """
    
    # Processor metadata
    KEY = "phash"
    VERSION = 1  # Increment if algorithm changes
    
    def __init__(self, hash_size: int = 16):
        """Initialize processor.
        
        Args:
            hash_size: Hash size for imagehash (default 16 = 256 bits)
        """
        self._hash_size = hash_size
    
    @property
    def key(self) -> str:
        """Unique identifier: 'phash'."""
        return self.KEY
    
    @property
    def version(self) -> int:
        """Version for cache invalidation."""
        return self.VERSION
    
    @property
    def depends_on(self) -> list[str]:
        """No dependencies - hash is computed first."""
        return []
    
    def can_process(self, ctx: 'ProcessingContext') -> bool:
        """Can process both images and videos.
        
        For images: Computes perceptual hash
        For videos: Computes file hash
        """
        # Can always process - different methods for images vs videos
        return True
    
    def process(self, ctx: 'ProcessingContext') -> str:
        """Compute hash for the file.
        
        Args:
            ctx: Processing context with image or video info
            
        Returns:
            Hash string like "phash:abc123" or "file:def456"
        """
        if ctx.is_video:
            return self._hash_video(ctx.path)
        elif ctx.image is not None:
            return self._hash_image(ctx.image)
        else:
            # Fallback to file hash if image couldn't be loaded
            return self._hash_file(ctx.path)
    
    def _hash_image(self, image: Image.Image) -> str:
        """Compute perceptual hash for PIL Image.
        
        Uses pHash algorithm which is robust to transformations.
        """
        try:
            phash = imagehash.phash(image, hash_size=self._hash_size)
            return f"phash:{str(phash)}"
        except Exception as e:
            # Fallback if pHash fails
            raise RuntimeError(f"Failed to compute perceptual hash: {e}")
    
    def _hash_video(self, path: Path) -> str:
        """Compute file content hash for videos.
        
        Uses xxhash for speed, falls back to md5.
        Hashes first 64KB + last 64KB + file size for speed.
        """
        try:
            import xxhash
            hasher = xxhash.xxh64()
        except ImportError:
            import hashlib
            hasher = hashlib.md5()
        
        try:
            file_size = path.stat().st_size
            chunk_size = 64 * 1024  # 64KB
            
            with open(path, 'rb') as f:
                # Hash first chunk
                hasher.update(f.read(chunk_size))
                
                # Hash last chunk if file is large enough
                if file_size > chunk_size * 2:
                    f.seek(-chunk_size, 2)
                    hasher.update(f.read(chunk_size))
                
                # Include file size to differentiate similar-start files
                hasher.update(str(file_size).encode())
            
            return f"file:{hasher.hexdigest()}"
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute file hash: {e}")
    
    def _hash_file(self, path: Path) -> str:
        """Fallback file hash when image can't be loaded."""
        return self._hash_video(path)  # Same logic


class VideoHashProcessor:
    """Computes file hash for videos only.
    
    Separate processor for videos that:
    - Skips images
    - Uses fast file hashing (xxhash)
    - Could be extended to use frame sampling in future
    """
    
    KEY = "vhash"
    VERSION = 1
    
    @property
    def key(self) -> str:
        return self.KEY
    
    @property
    def version(self) -> int:
        return self.VERSION
    
    @property
    def depends_on(self) -> list[str]:
        return []
    
    def can_process(self, ctx: 'ProcessingContext') -> bool:
        """Only process videos."""
        return ctx.is_video
    
    def process(self, ctx: 'ProcessingContext') -> str:
        """Compute file hash."""
        try:
            import xxhash
            hasher = xxhash.xxh64()
        except ImportError:
            import hashlib
            hasher = hashlib.md5()
        
        file_size = ctx.path.stat().st_size
        chunk_size = 64 * 1024
        
        with open(ctx.path, 'rb') as f:
            hasher.update(f.read(chunk_size))
            if file_size > chunk_size * 2:
                f.seek(-chunk_size, 2)
                hasher.update(f.read(chunk_size))
            hasher.update(str(file_size).encode())
        
        return f"vhash:{hasher.hexdigest()}"
