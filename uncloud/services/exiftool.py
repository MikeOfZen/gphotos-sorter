"""ExifTool service - centralized metadata operations.

This service manages ExifTool daemon instances for efficient metadata
read/write operations. Uses a thread-local daemon pool for thread safety.
"""
import atexit
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional

from uncloud.engines.metadata import ExifToolDaemon, ThreadLocalExifToolDaemon

logger = logging.getLogger(__name__)


class ExifToolService:
    """Centralized service for ExifTool metadata operations.
    
    This service:
    - Manages a thread-local pool of ExifTool daemons
    - Provides a simple API for metadata operations
    - Handles lifecycle (startup, shutdown)
    
    Usage:
        # At command startup
        exiftool = ExifToolService()
        
        # Read metadata
        subjects = exiftool.read_subjects(path)
        
        # Write metadata
        exiftool.write_subject(path, "uncloud:caption:1:\"a dog\"")
        
        # At shutdown
        exiftool.shutdown()
    """
    
    def __init__(self, batch_size: int = 50):
        """Initialize ExifTool service.
        
        Args:
            batch_size: Number of operations before flushing to disk
        """
        self._pool = ThreadLocalExifToolDaemon()
        self._batch_size = batch_size
        self._started = False
        
        # Register cleanup on exit
        atexit.register(self._cleanup)
        
        logger.debug("ExifToolService initialized")
    
    def _cleanup(self) -> None:
        """Cleanup handler for atexit."""
        if self._started:
            try:
                self.shutdown()
            except Exception:
                pass
    
    @property
    def daemon(self) -> ExifToolDaemon:
        """Get the current thread's ExifTool daemon.
        
        Returns:
            ExifToolDaemon instance for this thread
        """
        self._started = True
        return self._pool.get_daemon()
    
    def read_subjects(self, path: Path) -> list[str]:
        """Read XMP:Subject tags from file.
        
        Args:
            path: Path to media file
            
        Returns:
            List of subject strings
        """
        return self.daemon.extract_subjects(path)
    
    def write_subject(self, path: Path, subject: str) -> bool:
        """Write a subject tag to file.
        
        Args:
            path: Path to media file
            subject: Subject string to add
            
        Returns:
            True if successful
        """
        return self.daemon.write_subject(path, subject)
    
    def extract_metadata(
        self, 
        path: Path, 
        tags: Optional[list[str]] = None,
    ) -> dict:
        """Extract metadata from file.
        
        Args:
            path: Path to media file
            tags: Optional list of specific tags to extract
            
        Returns:
            Dictionary of metadata values
        """
        return self.daemon.extract(path, tags)
    
    def write_metadata(self, path: Path, metadata: dict) -> bool:
        """Write metadata to file.
        
        Args:
            path: Path to media file
            metadata: Dictionary of tag -> value
            
        Returns:
            True if successful
        """
        return self.daemon.write(path, metadata)
    
    def shutdown(self) -> None:
        """Shutdown all daemon instances."""
        if self._started:
            self._pool.close_all()
            self._started = False
            logger.debug("ExifToolService shut down")
    
    def __enter__(self) -> "ExifToolService":
        return self
    
    def __exit__(self, *args) -> None:
        self.shutdown()


# Global singleton
_exiftool_service: Optional[ExifToolService] = None


def get_exiftool_service() -> ExifToolService:
    """Get or create the global ExifTool service.
    
    Returns:
        The global ExifToolService instance
    """
    global _exiftool_service
    if _exiftool_service is None:
        _exiftool_service = ExifToolService()
    return _exiftool_service


def reset_exiftool_service() -> None:
    """Reset the global ExifTool service (for testing)."""
    global _exiftool_service
    if _exiftool_service is not None:
        _exiftool_service.shutdown()
        _exiftool_service = None
