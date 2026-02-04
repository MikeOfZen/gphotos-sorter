"""Core domain models and protocols."""
from .protocols import (
    HashEngine,
    MetadataExtractor,
    MediaRepository,
    ProgressReporter,
    FileOperations,
)
from .models import (
    MediaItem,
    HashResult,
    ProcessingResult,
    DuplicateGroup,
    CopyPlan,
)
from .config import SorterConfig, InputSource, OutputLayout

__all__ = [
    # Protocols
    "HashEngine",
    "MetadataExtractor",
    "MediaRepository",
    "ProgressReporter",
    "FileOperations",
    # Models
    "MediaItem",
    "HashResult",
    "ProcessingResult",
    "DuplicateGroup",
    "CopyPlan",
    # Config
    "SorterConfig",
    "InputSource",
    "OutputLayout",
]
