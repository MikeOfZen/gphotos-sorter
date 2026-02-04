"""Processing pipeline for multi-stage file analysis.

Runs a sequence of processors on each file:
- Hash computation
- CLIP embeddings (future)
- Face detection (future)
- Object detection (future)

Each processor:
- Reads cached result from file metadata if available
- Computes result if not cached
- Writes result to file metadata (optional)
- Stores result in database

All metadata uses 'uncloud:' prefix and includes version for invalidation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ..core.protocols import FileProcessor, ProgressReporter


# Media extensions for type detection
IMAGE_EXTENSIONS = frozenset({
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic', '.heif', '.tiff', '.tif'
})
VIDEO_EXTENSIONS = frozenset({
    '.mp4', '.mov', '.avi', '.mkv', '.m4v', '.3gp', '.wmv', '.webm'
})


@dataclass
class MediaData:
    """Loaded media file - images have PIL Image, videos don't.
    
    Videos are processed at file level only (hash of file content),
    not frame-by-frame analysis.
    """
    path: Path
    image: Optional[Image.Image]
    is_video: bool
    _loaded: bool = False
    
    @classmethod
    def load(cls, path: Path, progress: Optional[ProgressReporter] = None) -> 'MediaData':
        """Load media file.
        
        For images: Opens and loads PIL Image
        For videos: Just marks as video, no frame loading
        """
        suffix = path.suffix.lower()
        
        if suffix in VIDEO_EXTENSIONS:
            if progress:
                progress.debug(f"MediaData.load: {path.name} is video, skipping image load")
            return cls(path=path, image=None, is_video=True, _loaded=True)
        
        if suffix in IMAGE_EXTENSIONS:
            try:
                if progress:
                    progress.debug(f"MediaData.load: Loading image {path.name}")
                image = Image.open(path)
                # Load into memory to prevent file handle issues
                image.load()
                return cls(path=path, image=image, is_video=False, _loaded=True)
            except Exception as e:
                if progress:
                    progress.warning(f"MediaData.load: Failed to load {path.name}: {e}")
                return cls(path=path, image=None, is_video=False, _loaded=False)
        
        # Unknown extension
        if progress:
            progress.debug(f"MediaData.load: Unknown extension {suffix} for {path.name}")
        return cls(path=path, image=None, is_video=False, _loaded=False)


@dataclass
class ProcessingContextImpl:
    """Concrete implementation of ProcessingContext.
    
    Passed to each processor during pipeline execution.
    """
    path: Path
    media: MediaData
    results: dict[str, Any] = field(default_factory=dict)
    write_cache: bool = True
    _progress: Optional[ProgressReporter] = None
    
    @property
    def image(self) -> Optional[Image.Image]:
        """Get PIL Image, or None for videos."""
        return self.media.image
    
    @property
    def is_video(self) -> bool:
        """Whether this is a video file."""
        return self.media.is_video


class MetadataService:
    """Service for reading/writing processor results in file metadata.
    
    All metadata is stored with 'uncloud:' prefix in XMP:Subject field.
    Format: uncloud:<key>:<version>:<json_value>
    
    Example: uncloud:phash:1:"abc123def456"
    """
    
    # Prefix for all uncloud metadata
    PREFIX = "uncloud:"
    
    def __init__(self, daemon_pool: 'ThreadLocalExifToolDaemon', progress: Optional[ProgressReporter] = None):
        """Initialize with shared ExifTool daemon pool.
        
        Args:
            daemon_pool: Thread-local ExifTool daemons for metadata access
            progress: Optional progress reporter for debug logging
        """
        self._daemon_pool = daemon_pool
        self._progress = progress
    
    def _make_tag(self, key: str, version: int, value: Any) -> str:
        """Create metadata tag string.
        
        Format: uncloud:<key>:<version>:<json_value>
        """
        json_value = json.dumps(value, separators=(',', ':'))
        return f"{self.PREFIX}{key}:{version}:{json_value}"
    
    def _parse_tag(self, tag: str, key: str) -> Optional[tuple[int, Any]]:
        """Parse metadata tag string.
        
        Returns (version, value) or None if not matching.
        """
        expected_prefix = f"{self.PREFIX}{key}:"
        if not tag.startswith(expected_prefix):
            return None
        
        remainder = tag[len(expected_prefix):]
        try:
            # Format: <version>:<json_value>
            colon_idx = remainder.index(':')
            version = int(remainder[:colon_idx])
            json_value = remainder[colon_idx + 1:]
            value = json.loads(json_value)
            return version, value
        except (ValueError, json.JSONDecodeError):
            return None
    
    def read_cached(
        self, 
        file_path: Path, 
        key: str, 
        min_version: int
    ) -> Optional[Any]:
        """Read cached result from file metadata.
        
        Args:
            file_path: Path to media file
            key: Processor key (e.g., 'phash')
            min_version: Minimum acceptable version (older = stale)
            
        Returns:
            Cached value if found and version >= min_version, else None
        """
        if self._progress:
            self._progress.debug(f"MetadataService.read_cached: {file_path.name} key={key} min_v={min_version}")
        
        try:
            daemon = self._daemon_pool.get_daemon()
            if not daemon.is_alive:
                if self._progress:
                    self._progress.debug(f"MetadataService.read_cached: daemon not alive")
                return None
            
            # Read all XMP:Subject tags
            subjects = daemon.extract_subjects(file_path)
            if not subjects:
                if self._progress:
                    self._progress.debug(f"MetadataService.read_cached: no subjects found")
                return None
            
            # Find matching tag
            for subject in subjects:
                parsed = self._parse_tag(subject, key)
                if parsed:
                    version, value = parsed
                    if version >= min_version:
                        if self._progress:
                            self._progress.debug(f"MetadataService.read_cached: found v{version} for {key}")
                        return value
                    else:
                        if self._progress:
                            self._progress.debug(f"MetadataService.read_cached: stale v{version} < {min_version}")
            
            return None
            
        except Exception as e:
            if self._progress:
                self._progress.debug(f"MetadataService.read_cached: error: {e}")
            return None
    
    def write_cached(
        self,
        file_path: Path,
        key: str,
        version: int,
        value: Any,
    ) -> bool:
        """Write result to file metadata.
        
        Args:
            file_path: Path to media file
            key: Processor key
            version: Processor version
            value: Value to cache (must be JSON-serializable)
            
        Returns:
            True if successful
        """
        if self._progress:
            self._progress.debug(f"MetadataService.write_cached: {file_path.name} key={key} v={version}")
        
        try:
            daemon = self._daemon_pool.get_daemon()
            if not daemon.is_alive:
                if self._progress:
                    self._progress.debug(f"MetadataService.write_cached: daemon not alive")
                return False
            
            tag = self._make_tag(key, version, value)
            success = daemon.write_subject(file_path, tag)
            
            if self._progress:
                self._progress.debug(f"MetadataService.write_cached: {'success' if success else 'failed'}")
            
            return success
            
        except Exception as e:
            if self._progress:
                self._progress.debug(f"MetadataService.write_cached: error: {e}")
            return False


@dataclass
class PipelineStats:
    """Statistics from pipeline processing."""
    total_files: int = 0
    processed: int = 0
    skipped_video: int = 0
    errors: int = 0
    # Per-processor stats: key -> {from_cache, computed, written}
    processor_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    
    def init_processor(self, key: str) -> None:
        """Initialize stats for a processor."""
        if key not in self.processor_stats:
            self.processor_stats[key] = {
                'from_cache': 0,
                'computed': 0,
                'written': 0,
                'errors': 0,
            }
    
    def record_cache_hit(self, key: str) -> None:
        """Record a cache hit for processor."""
        self.processor_stats[key]['from_cache'] += 1
    
    def record_computed(self, key: str, written: bool) -> None:
        """Record a computed result."""
        self.processor_stats[key]['computed'] += 1
        if written:
            self.processor_stats[key]['written'] += 1
    
    def record_error(self, key: str) -> None:
        """Record a processor error."""
        self.processor_stats[key]['errors'] += 1


class ProcessingPipeline:
    """Pipeline that runs multiple processors on each file.
    
    Features:
    - Topological sort based on processor dependencies
    - Cache checking with version validation
    - Debug logging throughout
    - Per-processor statistics
    - Graceful error handling (one processor failing doesn't stop others)
    """
    
    def __init__(
        self,
        processors: list[FileProcessor],
        metadata_service: MetadataService,
        progress: ProgressReporter,
        write_cache: bool = True,
    ):
        """Initialize pipeline.
        
        Args:
            processors: List of processors to run
            metadata_service: Service for cache read/write
            progress: Progress reporter for logging
            write_cache: Whether to write results to file metadata
        """
        self._processors = self._sort_by_dependencies(processors, progress)
        self._metadata = metadata_service
        self._progress = progress
        self._write_cache = write_cache
        
        self._progress.debug(f"ProcessingPipeline: initialized with {len(processors)} processors")
        for p in self._processors:
            self._progress.debug(f"  - {p.key} v{p.version} depends_on={p.depends_on}")
    
    def _sort_by_dependencies(
        self, 
        processors: list[FileProcessor],
        progress: ProgressReporter,
    ) -> list[FileProcessor]:
        """Sort processors by dependencies (topological sort).
        
        Processors with no dependencies come first.
        """
        progress.debug("ProcessingPipeline: sorting processors by dependencies")
        
        # Build dependency graph
        by_key = {p.key: p for p in processors}
        result = []
        seen = set()
        
        def visit(p: FileProcessor):
            if p.key in seen:
                return
            seen.add(p.key)
            
            # Visit dependencies first
            for dep_key in p.depends_on:
                if dep_key in by_key:
                    visit(by_key[dep_key])
                else:
                    progress.warning(f"ProcessingPipeline: {p.key} depends on unknown {dep_key}")
            
            result.append(p)
        
        for p in processors:
            visit(p)
        
        progress.debug(f"ProcessingPipeline: sorted order: {[p.key for p in result]}")
        return result
    
    def process_file(self, file_path: Path, stats: PipelineStats) -> dict[str, Any]:
        """Run all processors on a single file.
        
        Args:
            file_path: Path to media file
            stats: Statistics to update
            
        Returns:
            Dict of processor_key -> result
        """
        self._progress.debug(f"ProcessingPipeline.process_file: {file_path.name}")
        
        # Load media
        media = MediaData.load(file_path, self._progress)
        
        # Create context
        ctx = ProcessingContextImpl(
            path=file_path,
            media=media,
            results={},
            write_cache=self._write_cache,
            _progress=self._progress,
        )
        
        # Run each processor
        for processor in self._processors:
            stats.init_processor(processor.key)
            
            # Check if processor can handle this file
            if not processor.can_process(ctx):
                self._progress.debug(f"  {processor.key}: skipping (can_process=False)")
                continue
            
            # Check dependencies
            missing_deps = [d for d in processor.depends_on if d not in ctx.results]
            if missing_deps:
                self._progress.debug(f"  {processor.key}: skipping (missing deps: {missing_deps})")
                continue
            
            # Try cache
            cached = self._metadata.read_cached(file_path, processor.key, processor.version)
            if cached is not None:
                self._progress.debug(f"  {processor.key}: cache hit")
                ctx.results[processor.key] = cached
                stats.record_cache_hit(processor.key)
                continue
            
            # Process
            self._progress.debug(f"  {processor.key}: computing...")
            try:
                result = processor.process(ctx)
                ctx.results[processor.key] = result
                self._progress.debug(f"  {processor.key}: computed successfully")
                
                # Write to cache
                written = False
                if self._write_cache:
                    try:
                        written = self._metadata.write_cached(
                            file_path, processor.key, processor.version, result
                        )
                    except Exception as e:
                        self._progress.debug(f"  {processor.key}: cache write failed: {e}")
                
                stats.record_computed(processor.key, written)
                
            except Exception as e:
                self._progress.debug(f"  {processor.key}: error: {e}")
                stats.record_error(processor.key)
                # Continue with next processor
        
        stats.processed += 1
        return ctx.results
    
    def print_stats(self, stats: PipelineStats) -> None:
        """Print detailed statistics."""
        self._progress.success(f"Processed {stats.processed}/{stats.total_files} files")
        
        for key, ps in stats.processor_stats.items():
            self._progress.info(
                f"  {key}: "
                f"{ps['from_cache']} cached, "
                f"{ps['computed']} computed, "
                f"{ps['written']} written, "
                f"{ps['errors']} errors"
            )
        
        if stats.skipped_video > 0:
            self._progress.info(f"  Skipped {stats.skipped_video} videos")
        if stats.errors > 0:
            self._progress.warning(f"  Total errors: {stats.errors}")


# Import here to avoid circular imports
from ..engines.metadata import ThreadLocalExifToolDaemon
