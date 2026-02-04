"""Service layer - business logic and shared services."""
from .scanner import DirectoryScanner
from .processor import MediaProcessor
from .deduplicator import DuplicateResolver
from .file_ops import FileManager
from .file_ops_sync import FileOpsSynchronizer, SyncResult
from .index_rebuilder import IndexRebuilder, RebuildStats

# New service layer
from .models import ModelService, get_model_service, reset_model_service
from .exiftool import ExifToolService, get_exiftool_service, reset_exiftool_service
from .app_context import AppContext, AppConfig, create_app_context
from .pipeline import ProcessingPipeline, MetadataService, PipelineStats
from .output import OutputService, OutputConfig, get_output_service, configure_output, reset_output_service

__all__ = [
    # Legacy services
    "DirectoryScanner",
    "MediaProcessor", 
    "DuplicateResolver",
    "FileManager",
    "FileOpsSynchronizer",
    "SyncResult",
    "IndexRebuilder",
    "RebuildStats",
    # Model service
    "ModelService",
    "get_model_service",
    "reset_model_service",
    # ExifTool service
    "ExifToolService",
    "get_exiftool_service",
    "reset_exiftool_service",
    # App context
    "AppContext",
    "AppConfig",
    "create_app_context",
    # Pipeline
    "ProcessingPipeline",
    "MetadataService",
    "PipelineStats",
    # Output service
    "OutputService",
    "OutputConfig",
    "get_output_service",
    "configure_output",
    "reset_output_service",
]

