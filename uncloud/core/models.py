"""Domain models - immutable data classes."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class ProcessingAction(Enum):
    """What action was taken on a file."""
    COPIED = "copied"
    SKIPPED_DUPLICATE = "skipped_duplicate"
    SKIPPED_KNOWN = "skipped_known"
    REPLACED = "replaced"
    ERROR = "error"


class DuplicateResolution(Enum):
    """How a duplicate was resolved."""
    KEEP_FIRST = "keep_first"
    KEEP_HIGHER_RESOLUTION = "keep_higher_resolution"
    KEEP_BOTH = "keep_both"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class MediaItem:
    """Represents a media file to be processed."""
    path: Path
    owner: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    sidecar_path: Optional[Path] = None
    is_media: bool = True
    
    @property
    def extension(self) -> str:
        return self.path.suffix.lower()
    
    @property
    def name(self) -> str:
        return self.path.name


@dataclass(frozen=True, slots=True)
class HashResult:
    """Result of hashing a media file."""
    item: MediaItem
    similarity_hash: Optional[str] = None
    date_taken: Optional[datetime] = None
    date_source: str = "unknown"
    width: Optional[int] = None
    height: Optional[int] = None
    error: Optional[str] = None
    
    @property
    def resolution(self) -> int:
        """Total pixels for comparison."""
        if self.width and self.height:
            return self.width * self.height
        return 0
    
    @property
    def is_valid(self) -> bool:
        return self.similarity_hash is not None and self.error is None


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """A group of files with the same hash."""
    hash_value: str
    items: tuple[HashResult, ...]
    
    @property
    def count(self) -> int:
        return len(self.items)
    
    def get_best(self, strategy: DuplicateResolution) -> HashResult:
        """Get the best item according to strategy."""
        if strategy == DuplicateResolution.KEEP_HIGHER_RESOLUTION:
            return max(self.items, key=lambda x: x.resolution)
        return self.items[0]  # KEEP_FIRST


@dataclass(frozen=True, slots=True)
class CopyPlan:
    """Plan for copying a file."""
    source: MediaItem
    target_dir: Path
    target_name: str
    hash_result: HashResult
    replace_existing: Optional[Path] = None  # Path to replace if duplicate
    
    @property
    def target_path(self) -> Path:
        return self.target_dir / self.target_name


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Result of processing a single file."""
    item: MediaItem
    action: ProcessingAction
    target_path: Optional[Path] = None
    replaced_path: Optional[Path] = None
    error: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        return self.action != ProcessingAction.ERROR


@dataclass(slots=True)
class ProcessingStats:
    """Mutable statistics for a processing run."""
    total_files: int = 0
    processed: int = 0
    copied: int = 0
    skipped_known: int = 0
    skipped_duplicate: int = 0
    replaced: int = 0
    errors: int = 0
    non_media: int = 0
    hash_collisions: int = 0
    elapsed_seconds: float = 0.0
    
    # Aliases for backward compatibility
    @property
    def files_scanned(self) -> int:
        return self.total_files
    
    @property
    def files_copied(self) -> int:
        return self.copied
    
    @property
    def duplicates_skipped(self) -> int:
        return self.skipped_duplicate
    
    def record(self, result: ProcessingResult) -> None:
        """Record a processing result."""
        self.processed += 1
        match result.action:
            case ProcessingAction.COPIED:
                self.copied += 1
            case ProcessingAction.SKIPPED_DUPLICATE:
                self.skipped_duplicate += 1
            case ProcessingAction.SKIPPED_KNOWN:
                self.skipped_known += 1
            case ProcessingAction.REPLACED:
                self.replaced += 1
            case ProcessingAction.ERROR:
                self.errors += 1
    
    def summary(self) -> dict[str, int]:
        return {
            "total": self.total_files,
            "processed": self.processed,
            "copied": self.copied,
            "skipped_known": self.skipped_known,
            "skipped_duplicate": self.skipped_duplicate,
            "replaced": self.replaced,
            "errors": self.errors,
            "non_media": self.non_media,
        }
