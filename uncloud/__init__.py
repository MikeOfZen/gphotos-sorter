"""Media ingestion and organization package.

New architecture with dependency injection and clean interfaces.
"""

__version__ = "2.0.0"

# Core exports
from .core.config import SorterConfig, InputSource, OutputLayout, DuplicatePolicy
from .core.models import MediaItem, HashResult, ProcessingResult, ProcessingStats
from .core.protocols import HashEngine, MetadataExtractor, MediaRepository, ProgressReporter

# Engine exports
from .engines.hash_engine import CPUHashEngine, create_hash_engine
from .engines.metadata import ExifToolMetadataExtractor

# Service exports
from .services.processor import MediaProcessor, ProcessorDependencies
from .services.scanner import DirectoryScanner
from .services.deduplicator import DuplicateResolver
from .services.file_ops import FileManager

# Persistence exports
from .persistence.database import SQLiteMediaRepository

# Logging exports
from .logging.rich_logger import RichProgressReporter

__all__ = [
    # Core
    "SorterConfig",
    "InputSource",
    "OutputLayout",
    "DuplicatePolicy",
    "MediaItem",
    "HashResult",
    "ProcessingResult",
    "ProcessingStats",
    "HashEngine",
    "MetadataExtractor",
    "MediaRepository",
    "ProgressReporter",
    # Engines
    "CPUHashEngine",
    "create_hash_engine",
    "ExifToolMetadataExtractor",
    # Services
    "MediaProcessor",
    "ProcessorDependencies",
    "DirectoryScanner",
    "DuplicateResolver",
    "FileManager",
    # Persistence
    "SQLiteMediaRepository",
    # Logging
    "RichProgressReporter",
]
