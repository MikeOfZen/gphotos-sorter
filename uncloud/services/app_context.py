"""Application context - unified service management for CLI commands.

This module provides a centralized way to set up and manage all services
needed by CLI commands. It ensures proper initialization order, provides
progress feedback to users, and handles cleanup.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console

from uncloud.persistence.database import SQLiteMediaRepository
from uncloud.services.exiftool import ExifToolService
from uncloud.services.models import ModelService
from uncloud.services.pipeline import MetadataService


logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Configuration for application context."""
    
    # Database
    db_path: Optional[Path] = None
    
    # Models to load
    models: list[str] = field(default_factory=list)
    
    # Device for AI models
    device: str = "auto"
    
    # Whether to write cache to file metadata
    write_cache: bool = True
    
    # Verbose output
    verbose: bool = False


class AppContext:
    """Application context managing all services for a CLI command.
    
    This class:
    - Initializes all services in correct order
    - Provides progress feedback during model loading
    - Ensures proper cleanup on exit
    - Provides easy access to all services
    
    Usage:
        with AppContext(config) as ctx:
            # All services are ready
            pipeline = ProcessingPipeline(
                processors=[CaptionProcessor(ctx.models)],
                metadata_service=ctx.metadata,
                progress=progress,
            )
            
            # Process files...
        
        # All services cleaned up automatically
    
    Or without context manager:
        ctx = AppContext(config)
        ctx.initialize(console)
        try:
            # Use services...
        finally:
            ctx.shutdown()
    """
    
    def __init__(self, config: AppConfig):
        """Initialize application context.
        
        Args:
            config: Application configuration
        """
        self._config = config
        self._console: Optional[Console] = None
        
        # Services (initialized lazily or in initialize())
        self._db: Optional[SQLiteMediaRepository] = None
        self._exiftool: Optional[ExifToolService] = None
        self._models: Optional[ModelService] = None
        self._metadata: Optional[MetadataService] = None
        
        self._initialized = False
    
    def initialize(self, console: Optional[Console] = None) -> None:
        """Initialize all services with progress feedback.
        
        Args:
            console: Rich console for progress output
        """
        if self._initialized:
            return
        
        self._console = console or Console(stderr=True)
        
        def log(msg: str):
            if self._console:
                self._console.print(msg)
            logger.info(msg.replace('[', '').replace(']', ''))
        
        log("[bold cyan]Initializing services...[/bold cyan]")
        
        # 1. Database (if path provided)
        if self._config.db_path:
            log(f"  [cyan]• Database:[/cyan] {self._config.db_path}")
            self._db = SQLiteMediaRepository(self._config.db_path)
        
        # 2. ExifTool daemon pool
        log("  [cyan]• ExifTool daemon pool[/cyan]")
        self._exiftool = ExifToolService()
        
        # 3. Metadata service (uses exiftool)
        from uncloud.engines.metadata import ThreadLocalExifToolDaemon
        # Create a wrapper that works with MetadataService
        daemon_pool = self._exiftool._pool
        self._metadata = MetadataService(daemon_pool)
        
        # 4. AI Models (if any requested)
        if self._config.models:
            self._models = ModelService(device=self._config.device)
            self._models.load_models(
                self._config.models, 
                progress=log if self._config.verbose else None,
            )
        
        self._initialized = True
        log("[bold green]✓ Services ready[/bold green]\n")
    
    @property
    def db(self) -> Optional[SQLiteMediaRepository]:
        """Get database repository."""
        return self._db
    
    @property
    def exiftool(self) -> ExifToolService:
        """Get ExifTool service."""
        if self._exiftool is None:
            self._exiftool = ExifToolService()
        return self._exiftool
    
    @property
    def models(self) -> Optional[ModelService]:
        """Get model service."""
        return self._models
    
    @property
    def metadata(self) -> MetadataService:
        """Get metadata service for pipeline caching."""
        if self._metadata is None:
            from uncloud.engines.metadata import ThreadLocalExifToolDaemon
            daemon_pool = ThreadLocalExifToolDaemon()
            self._metadata = MetadataService(daemon_pool)
        return self._metadata
    
    @property
    def config(self) -> AppConfig:
        """Get application configuration."""
        return self._config
    
    def shutdown(self) -> None:
        """Shutdown all services."""
        if self._models:
            self._models.unload_all()
            self._models = None
        
        if self._exiftool:
            self._exiftool.shutdown()
            self._exiftool = None
        
        if self._db:
            self._db.close()
            self._db = None
        
        self._initialized = False
        logger.debug("AppContext shut down")
    
    def __enter__(self) -> "AppContext":
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, *args) -> None:
        self.shutdown()


def create_app_context(
    db_path: Optional[Path] = None,
    models: Optional[list[str]] = None,
    device: str = "auto",
    write_cache: bool = True,
    verbose: bool = False,
) -> AppContext:
    """Create an application context with specified configuration.
    
    Convenience function for creating AppContext with common options.
    
    Args:
        db_path: Path to SQLite database
        models: List of AI models to load (e.g., ["caption", "faces"])
        device: Device for AI models ("auto", "cuda", "cpu")
        write_cache: Whether to write results to file metadata
        verbose: Enable verbose output
        
    Returns:
        Configured AppContext instance
    """
    config = AppConfig(
        db_path=db_path,
        models=models or [],
        device=device,
        write_cache=write_cache,
        verbose=verbose,
    )
    return AppContext(config)
