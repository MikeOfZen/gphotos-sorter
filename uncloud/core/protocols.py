"""Protocol definitions (interfaces) for dependency injection."""
from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Protocol, Optional, Iterator, Any

from .models import MediaItem


class HashEngine(Protocol):
    """Interface for computing perceptual/content hashes.
    
    Implementations:
    - CPUHashEngine: Uses imagehash (slow but universal)
    - GPUHashEngine: Uses CUDA/OpenCL for massive speedup
    """
    
    @abstractmethod
    def compute_hash(self, path: Path) -> Optional[str]:
        """Compute hash for a single file."""
        ...
    
    @abstractmethod
    def compute_batch(self, paths: list[Path]) -> list[Optional[str]]:
        """Compute hashes for multiple files (enables GPU batching)."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name for logging."""
        ...
    
    @property
    @abstractmethod
    def supports_gpu(self) -> bool:
        """Whether this engine can use GPU acceleration."""
        ...


class MetadataExtractor(Protocol):
    """Interface for extracting/writing file metadata."""
    
    @abstractmethod
    def extract_datetime(self, path: Path, sidecar: Optional[Path] = None) -> tuple[Optional[datetime], str]:
        """Extract date taken from file or sidecar. Returns (datetime, source)."""
        ...
    
    @abstractmethod
    def extract_resolution(self, path: Path) -> tuple[Optional[int], Optional[int]]:
        """Extract image dimensions. Returns (width, height)."""
        ...
    
    @abstractmethod
    def write_tags(self, path: Path, tags: dict[str, Any]) -> bool:
        """Write EXIF tags to file."""
        ...
    
    @abstractmethod
    def flush(self) -> bool:
        """Flush any pending writes. Returns True if successful."""
        ...
    
    @abstractmethod
    def close(self) -> None:
        """Clean up resources (e.g., exiftool process)."""
        ...


class MediaRepository(Protocol):
    """Interface for media database operations."""
    
    @abstractmethod
    def get_by_hash(self, similarity_hash: str) -> Optional[Any]:
        """Get record by hash."""
        ...
    
    @abstractmethod
    def has_source_path(self, path: str) -> bool:
        """Check if source path already processed."""
        ...
    
    @abstractmethod
    def get_all_source_paths(self) -> set[str]:
        """Get all known source paths."""
        ...
    
    @abstractmethod
    def upsert(self, record: Any) -> None:
        """Insert or update a record."""
        ...
    
    @abstractmethod
    def add_pending_operation(self, source: str, target: str, hash_val: str, op: str) -> int:
        """Track a pending operation for crash recovery."""
        ...
    
    @abstractmethod
    def complete_pending_operation(self, op_id: int) -> None:
        """Mark operation as complete."""
        ...
    
    @abstractmethod
    def get_pending_operations(self) -> list[Any]:
        """Get all pending operations."""
        ...
    
    @abstractmethod
    def clear_all_pending_operations(self) -> int:
        """Clear all pending operations."""
        ...
    
    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        ...


class ProgressReporter(Protocol):
    """Interface for progress reporting."""
    
    @abstractmethod
    def start_phase(self, name: str, total: int) -> None:
        """Start a new processing phase."""
        ...
    
    @abstractmethod
    def update_phase(self, completed: int, description: Optional[str] = None) -> None:
        """Update progress."""
        ...
    
    @abstractmethod
    def advance_phase(self, amount: int = 1) -> None:
        """Advance the current phase by an amount."""
        ...
    
    @abstractmethod
    def end_phase(self) -> None:
        """Complete current phase."""
        ...
    
    @abstractmethod
    def info(self, message: str) -> None:
        """Log an info message."""
        ...
    
    @abstractmethod
    def warning(self, message: str) -> None:
        """Log a warning message."""
        ...
    
    @abstractmethod
    def error(self, message: str) -> None:
        """Log an error message."""
        ...


class FileOperations(Protocol):
    """Interface for file operations."""
    
    @abstractmethod
    def copy_file(self, source: Path, target: Path) -> bool:
        """Copy a file."""
        ...
    
    @abstractmethod
    def move_file(self, source: Path, target: Path) -> bool:
        """Move a file."""
        ...
    
    @abstractmethod
    def delete_file(self, path: Path) -> bool:
        """Delete a file."""
        ...
    
    @abstractmethod
    def ensure_directory(self, path: Path) -> None:
        """Ensure directory exists."""
        ...
    
    @abstractmethod
    def find_unique_path(self, directory: Path, basename: str, extension: str) -> Path:
        """Find a unique filename in directory."""
        ...


class MediaScanner(Protocol):
    """Interface for scanning directories for media files."""
    
    @abstractmethod
    def scan(
        self,
        sources: list[Any],  # InputSource
        recursive: bool = True,
        skip_known: Optional[set[str]] = None,
    ) -> Iterator[MediaItem]:
        """Scan sources and yield MediaItems."""
        ...
    
    @abstractmethod
    def count_files(
        self,
        sources: list[Any],
        recursive: bool = True,
    ) -> tuple[int, int]:  # (media_count, non_media_count)
        """Count files without loading them all."""
        ...


# ============ Processing Pipeline Protocols ============

class ProcessingContext(Protocol):
    """Context passed to each processor during pipeline execution."""
    
    @property
    def path(self) -> Path:
        """Path to the media file being processed."""
        ...
    
    @property
    def image(self) -> Optional[Any]:  # PIL.Image.Image
        """Loaded PIL Image for images, None for videos."""
        ...
    
    @property
    def is_video(self) -> bool:
        """Whether this file is a video (image processing skipped)."""
        ...
    
    @property
    def results(self) -> dict[str, Any]:
        """Results from earlier processors in the pipeline.
        
        Keys are processor keys like 'phash_v1'.
        Used for processor dependencies.
        """
        ...
    
    @property
    def write_cache(self) -> bool:
        """Whether to write results to file metadata."""
        ...


class FileProcessor(Protocol):
    """Interface for file processing operators.
    
    Each processor extracts one piece of information from a file:
    - Perceptual hash
    - CLIP embedding
    - Face detection
    - Object detection
    - etc.
    
    Results are cached in file metadata (with uncloud: prefix)
    and stored in database.
    """
    
    @property
    def key(self) -> str:
        """Unique identifier for this processor.
        
        Used for:
        - Cache key in metadata (prefixed with 'uncloud:')
        - Column/field in database
        - Dependency resolution
        - Error reporting
        
        Examples: 'phash', 'clip', 'faces', 'objects'
        Format: lowercase, no version (version is separate property)
        """
        ...
    
    @property
    def version(self) -> int:
        """Version number for cache invalidation.
        
        Increment when:
        - Algorithm changes
        - Model updates
        - Output format changes
        
        Cached values with older versions are recomputed.
        """
        ...
    
    @property
    def depends_on(self) -> list[str]:
        """Keys of processors that must run before this one.
        
        Empty list means no dependencies.
        
        Examples:
        - [] for phash (no dependencies)
        - ['faces'] for face_recognition (needs face bounding boxes)
        """
        ...
    
    def can_process(self, ctx: ProcessingContext) -> bool:
        """Check if this processor can handle this file.
        
        Use to skip:
        - Videos for image-only processors
        - Images for video-only processors
        - Unsupported file formats
        
        Args:
            ctx: Processing context with file info
            
        Returns:
            True if this processor should run on this file
        """
        ...
    
    def process(self, ctx: ProcessingContext) -> Any:
        """Process a file and return result.
        
        Args:
            ctx: Processing context containing:
                - ctx.path: File path
                - ctx.image: PIL Image (None for videos)
                - ctx.results: Results from earlier processors
        
        Returns:
            Computed result (any JSON-serializable type):
            - str for hash
            - list[float] for embeddings  
            - list[dict] for detected objects
            - etc.
        
        Raises:
            Exception: Processing failed (caught by pipeline)
        """
        ...
