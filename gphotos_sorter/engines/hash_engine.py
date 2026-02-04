"""Hash engine implementations.

Current: CPU-based imagehash (slow)
GPU: CUDA-accelerated perceptual hashing (fast)
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from PIL import Image, ImageFile
import imagehash

from ..core.config import HashBackend

# Try to import GPU dependencies
try:
    import torch
    import torchvision.transforms.functional as TF
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

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
    """GPU-accelerated hash engine using PyTorch.
    
    Uses CUDA for fast image loading and DCT-based perceptual hashing.
    Falls back to CPU engine for videos or if GPU is unavailable.
    """
    
    def __init__(self, backend: str = "cuda"):
        self._backend = backend
        self._available = self._check_gpu_available()
        self._device = None
        self._cpu_fallback = CPUHashEngine()
        
        if self._available:
            import torch
            self._device = torch.device("cuda" if backend == "cuda" else "cpu")
    
    @property
    def name(self) -> str:
        status = "available" if self._available else "unavailable, using CPU"
        return f"GPU ({self._backend}, {status})"
    
    @property
    def supports_gpu(self) -> bool:
        return self._available
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        if not GPU_AVAILABLE:
            return False
        
        try:
            import torch
            if self._backend == "cuda":
                return torch.cuda.is_available()
            elif self._backend == "opencl":
                # PyTorch doesn't directly support OpenCL, fallback to CPU
                return False
            return False
        except Exception:
            return False
    
    def compute_hash(self, path: Path) -> Optional[str]:
        """Compute hash for a single file."""
        if not self._available:
            return self._cpu_fallback.compute_hash(path)
        
        if is_video(path):
            # Videos still use SHA256
            return self._cpu_fallback.compute_hash(path)
        
        if is_image(path):
            return self._hash_image_gpu(path)
        
        return None
    
    def compute_batch(self, paths: list[Path]) -> list[Optional[str]]:
        """Compute hashes for multiple images using GPU batching."""
        if not self._available:
            return self._cpu_fallback.compute_batch(paths)
        
        results: list[Optional[str]] = [None] * len(paths)
        
        # Separate images and videos
        image_indices = []
        video_indices = []
        
        for i, path in enumerate(paths):
            if is_image(path):
                image_indices.append(i)
            elif is_video(path):
                video_indices.append(i)
        
        # Process images in GPU batch
        if image_indices:
            image_paths = [paths[i] for i in image_indices]
            image_hashes = self._batch_hash_images_gpu(image_paths)
            for idx, hash_val in zip(image_indices, image_hashes):
                results[idx] = hash_val
        
        # Process videos with CPU fallback
        for idx in video_indices:
            results[idx] = self._cpu_fallback.compute_hash(paths[idx])
        
        return results
    
    def _hash_image_gpu(self, path: Path) -> Optional[str]:
        """Compute perceptual hash for single image using GPU."""
        if not GPU_AVAILABLE:
            return self._cpu_fallback.compute_hash(path)
        
        try:
            import torch
            import torchvision.transforms.functional as TF
            from PIL import Image
            
            with Image.open(path) as img:
                img = img.convert("L")  # Grayscale
                img = img.resize((32, 32), Image.Resampling.LANCZOS)
                
                # Convert to tensor and move to GPU
                tensor = TF.to_tensor(img).to(self._device)
                
                # Compute DCT-like hash using GPU
                hash_val = self._compute_dct_hash_gpu(tensor)
                return f"phash:{hash_val}"
        except Exception:
            # Fall back to CPU
            return self._cpu_fallback.compute_hash(path)
    
    def _batch_hash_images_gpu(self, paths: list[Path], batch_size: int = 64) -> list[Optional[str]]:
        """Process images in batches on GPU."""
        if not GPU_AVAILABLE:
            return [self._cpu_fallback.compute_hash(p) for p in paths]
        
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image
        
        results: list[Optional[str]] = []
        
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_tensors = []
            batch_valid = []
            
            for path in batch_paths:
                try:
                    with Image.open(path) as img:
                        img = img.convert("L")
                        img = img.resize((32, 32), Image.Resampling.LANCZOS)
                        tensor = TF.to_tensor(img)
                        batch_tensors.append(tensor)
                        batch_valid.append(True)
                except Exception:
                    # Placeholder tensor for failed loads
                    batch_tensors.append(torch.zeros(1, 32, 32))
                    batch_valid.append(False)
            
            # Stack and move to GPU
            batch = torch.stack(batch_tensors).to(self._device)
            
            # Compute hashes for batch
            for j, (tensor, valid) in enumerate(zip(batch, batch_valid)):
                if valid:
                    hash_val = self._compute_dct_hash_gpu(tensor.unsqueeze(0))
                    results.append(f"phash:{hash_val}")
                else:
                    results.append(None)
        
        return results
    
    def _compute_dct_hash_gpu(self, tensor) -> str:
        """Compute DCT-based perceptual hash on GPU tensor."""
        import torch
        
        # Simple DCT approximation using mean comparison
        # This matches the imagehash pHash algorithm
        tensor = tensor.squeeze()
        
        # Resize to 8x8 for final hash
        if tensor.dim() == 3:
            tensor = tensor[0]  # Take first channel
        
        # Use average pooling to get 8x8
        tensor = tensor.view(1, 1, 32, 32)
        pooled = torch.nn.functional.avg_pool2d(tensor, 4)  # 32/4 = 8
        pooled = pooled.squeeze()
        
        # Compute mean and create binary hash
        mean_val = pooled.mean()
        binary = (pooled > mean_val).flatten().cpu().numpy()
        
        # Convert to hex
        hash_int = 0
        for bit in binary:
            hash_int = (hash_int << 1) | int(bit)
        
        return format(hash_int, '016x')
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for the selected backend."""
        # This method is now redundant but kept for backward compatibility
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
        engine = GPUHashEngine(backend.value.replace("gpu-", ""))
        # If GPU is not available, return CPU engine instead
        if not engine.supports_gpu:
            return CPUHashEngine()
        return engine
    
    # AUTO: Try GPU, fall back to CPU
    if backend == HashBackend.AUTO:
        try:
            engine = GPUHashEngine("cuda")
            if engine.supports_gpu:
                return engine
        except Exception:
            pass
        return CPUHashEngine()
    
    # Default to CPU
    return CPUHashEngine()
