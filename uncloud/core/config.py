"""Configuration dataclasses with validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class OutputLayout(Enum):
    """How to organize output files."""
    YEAR_MONTH = "year-month"      # 2024/2024-01/
    YEAR_MONTH_DAY = "year-month-day"  # 2024/2024-01/2024-01-15/
    FLAT = "flat"                  # All in one folder
    SINGLE = "single"              # Single folder per run


class DuplicatePolicy(Enum):
    """How to handle duplicates."""
    SKIP = "skip"
    KEEP_FIRST = "keep-first"
    KEEP_HIGHER_RESOLUTION = "keep-higher-resolution"
    KEEP_BOTH = "keep-both"
    # Aliases for CLI compatibility
    KEEP_LARGER = KEEP_HIGHER_RESOLUTION
    KEEP_SMALLER = "keep-smaller"  # Not really used but for CLI
    KEEP_ALL = KEEP_BOTH


class HashBackend(Enum):
    """Which hash engine to use."""
    AUTO = "auto"        # Auto-detect GPU availability
    CPU = "cpu"          # Force CPU (imagehash)
    GPU_CUDA = "cuda"    # Force CUDA
    GPU_OPENCL = "opencl"  # Force OpenCL


@dataclass(frozen=True, slots=True)
class InputSource:
    """An input directory with owner tag."""
    path: Path
    owner: str = "default"
    
    def __post_init__(self) -> None:
        if not self.path.exists():
            raise ValueError(f"Input path does not exist: {self.path}")
    
    @classmethod
    def from_dict(cls, data: dict) -> "InputSource":
        return cls(
            path=Path(data["path"]).expanduser().resolve(),
            owner=data.get("owner", "default"),
        )


@dataclass(frozen=True, slots=True)
class FilenameFormat:
    """How to format output filenames."""
    include_time: bool = True
    year_format: str = "YYYY"  # YYYY or YY
    month_format: str = "MM"   # MM, M, MMM, MMMM
    day_format: str = "DD"     # DD or D
    include_tags: bool = True
    max_tags: Optional[int] = 5


@dataclass(slots=True)
class SorterConfig:
    """Main configuration for the sorter.
    
    All fields are validated on construction.
    This is the only configuration object passed through the system.
    """
    # Required
    output_root: Path
    inputs: tuple[InputSource, ...]
    
    # Output organization
    layout: OutputLayout = OutputLayout.YEAR_MONTH
    filename_format: FilenameFormat = field(default_factory=FilenameFormat)
    
    # Processing options
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.KEEP_HIGHER_RESOLUTION
    copy_non_media: bool = False
    copy_sidecar: bool = True
    modify_exif: bool = True
    recursive: bool = True
    
    # Performance
    workers: int = 4
    hash_backend: HashBackend = HashBackend.AUTO
    batch_size: int = 100  # For GPU batching
    
    # Database
    db_path: Optional[Path] = None
    
    # Execution mode
    dry_run: bool = False
    limit: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.inputs:
            raise ValueError("At least one input source is required")
        
        if self.workers < 1:
            raise ValueError("Workers must be at least 1")
        
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        
        # Resolve db_path
        if self.db_path is None:
            object.__setattr__(self, "db_path", self.output_root / "media.sqlite")
        
        # Ensure output exists
        if not self.dry_run:
            self.output_root.mkdir(parents=True, exist_ok=True)
    
    @property
    def resolved_db_path(self) -> Path:
        return self.db_path or (self.output_root / "media.sqlite")
    
    def with_overrides(self, **kwargs) -> "SorterConfig":
        """Create a new config with some values overridden."""
        from dataclasses import asdict
        current = asdict(self)
        current.update(kwargs)
        return SorterConfig(**current)
