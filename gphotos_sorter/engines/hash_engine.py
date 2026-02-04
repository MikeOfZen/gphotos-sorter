"""Hash engine implementations.

Current: CPU-based imagehash (slow)
Future: GPU-accelerated perceptual hashing
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFile
import imagehash

from ..core.config import HashBackend

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", 
    ".tif", ".tiff", ".bmp", ".gif"
})

VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
    ".3gp", ".wmv", ".flv", ".mts", ".m2ts"
})


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_media(path: Path) -> bool:
    return is_image(path) or is_video(path)


class CPUHashEngine:
    """CPU-based hash engine using imagehash.
    
    This is the fallback/default engine. It's slow but works everywhere.
    Uses ThreadPoolExecutor for I/O-bound operations.
    """
    
    def __init__(self, max_image_pixels: int = 256 * 1024 * 1024 // 4):
        """Initialize the hash engine.
        
        Args:
            max_image_pixels: Maximum pixels for decompression bomb protection.
        """
        self._max_pixels = max_image_pixels
        Image.MAX_IMAGE_PIXELS = max_image_pixels
    
    @property
    def name(self) -> str:
        return "CPU (imagehash)"
    
    @property
    def supports_gpu(self) -> bool:
        return False
    
    def compute_hash(self, path: Path) -> Optional[str]:
        """Compute hash for a single file."""
        try:
            if is_image(path):
                return self._hash_image(path)
            elif is_video(path):
                return self._hash_video(path)
            return None
        except Exception:
            return None
    
    def compute_batch(self, paths: list[Path]) -> list[Optional[str]]:
        """Compute hashes for multiple files using thread pool.
        
        For CPU engine, this uses threads for I/O parallelism.
        For GPU engine (future), this would batch onto GPU.
        """
        results: list[Optional[str]] = [None] * len(paths)
        
        # Use threads for I/O-bound work
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_idx = {
                executor.submit(self.compute_hash, path): i 
                for i, path in enumerate(paths)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = None
        
        return results
    
    def _hash_image(self, path: Path) -> Optional[str]:
        """Compute perceptual hash for an image."""
        try:
            with Image.open(path) as img:
                # Force load to catch issues early
                img.load()
                # Compute perceptual hash
                phash = imagehash.phash(img)
                return f"phash:{phash}"
        except Exception:
            return None
    
    def _hash_video(self, path: Path) -> Optional[str]:
        """Compute SHA256 hash for a video file."""
        return f"sha256:{self._sha256_file(path)}"
    
    @staticmethod
    def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
        """Compute SHA256 of file contents."""
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()


class GPUHashEngine:
    """GPU-accelerated hash engine (placeholder for Phase 2).
    
    Will use CUDA or OpenCL for massive speedup on image processing.
    Potential libraries:
    - cupy + cucim for CUDA
    - PyOpenCL for OpenCL
    - torch with GPU for neural hash
    """
    
    def __init__(self, backend: str = "cuda"):
        self._backend = backend
        self._available = self._check_gpu_available()
    
    @property
    def name(self) -> str:
        return f"GPU ({self._backend})"
    
    @property
    def supports_gpu(self) -> bool:
        return self._available
    
    def compute_hash(self, path: Path) -> Optional[str]:
        # Placeholder - falls back to CPU
        return CPUHashEngine().compute_hash(path)
    
    def compute_batch(self, paths: list[Path]) -> list[Optional[str]]:
        # Placeholder - this is where GPU batching would happen
        # In Phase 2: Load images to GPU memory, compute hashes in parallel
        return CPUHashEngine().compute_batch(paths)
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for the selected backend."""
        if self._backend == "cuda":
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
        return False


def create_hash_engine(backend: HashBackend = HashBackend.AUTO) -> CPUHashEngine | GPUHashEngine:
    """Factory function to create the appropriate hash engine.
    
    Args:
        backend: Which backend to use. AUTO will try GPU first.
    
    Returns:
        A HashEngine implementation.
    """
    if backend == HashBackend.CPU:
        return CPUHashEngine()
    
    if backend in (HashBackend.GPU_CUDA, HashBackend.GPU_OPENCL):
        engine = GPUHashEngine(backend.value)
        if engine.supports_gpu:
            return engine
        # Fall back to CPU
        return CPUHashEngine()
    
    # AUTO: Try GPU, fall back to CPU
    if backend == HashBackend.AUTO:
        try:
            engine = GPUHashEngine("cuda")
            if engine.supports_gpu:
                return engine
        except Exception:
            pass
        return CPUHashEngine()
    
    return CPUHashEngine()
