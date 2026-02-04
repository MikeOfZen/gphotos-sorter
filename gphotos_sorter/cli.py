"""New CLI implementation with dependency injection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .core.config import (
    SorterConfig,
    InputSource,
    OutputLayout,
    DuplicatePolicy,
    HashBackend,
)
from .core.protocols import ProgressReporter
from .engines.hash_engine import create_hash_engine
from .engines.metadata import ExifToolMetadataExtractor
from .persistence.database import SQLiteMediaRepository
from .services.scanner import DirectoryScanner
from .services.deduplicator import DuplicateResolver
from .services.file_ops import FileManager
from .services.processor import MediaProcessor, ProcessorDependencies
from .logging.rich_logger import RichProgressReporter, QuietProgressReporter


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="gphotos-sorter",
        description="Sort and deduplicate Google Photos exports.",
    )
    
    # Input/Output
    parser.add_argument(
        "input_dirs",
        nargs="+",
        type=Path,
        help="Input directories (format: PATH or owner:PATH)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory",
    )
    
    # Layout options
    parser.add_argument(
        "--layout",
        type=str,
        choices=["year-month", "year-month-day", "flat", "single"],
        default="year-month",
        help="Output directory layout (default: year-month)",
    )
    
    # Duplicate handling
    parser.add_argument(
        "--duplicates",
        type=str,
        choices=["skip", "keep-larger", "keep-smaller", "keep-all"],
        default="skip",
        help="How to handle duplicates (default: skip)",
    )
    
    # Hash backend
    parser.add_argument(
        "--hash-backend",
        type=str,
        choices=["auto", "cpu", "gpu-cuda", "gpu-opencl"],
        default="auto",
        help="Hash computation backend (default: auto)",
    )
    
    # Processing options
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: CPU count)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )
    
    # Output options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-essential output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without copying",
    )
    
    # Database
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to database file (default: OUTPUT/.gphotos-sorter.db)",
    )
    
    return parser.parse_args(argv)


def parse_input_sources(input_dirs: list[Path]) -> tuple[InputSource, ...]:
    """Parse input directories with optional owner prefix."""
    sources = []
    for inp in input_dirs:
        inp_str = str(inp)
        if ":" in inp_str and not inp_str.startswith("/"):
            # Format: owner:path
            owner, path_str = inp_str.split(":", 1)
            sources.append(InputSource(path=Path(path_str), owner=owner))
        else:
            sources.append(InputSource(path=inp))
    return tuple(sources)


def layout_from_string(s: str) -> OutputLayout:
    """Convert string to OutputLayout enum."""
    mapping = {
        "year-month": OutputLayout.YEAR_MONTH,
        "year-month-day": OutputLayout.YEAR_MONTH_DAY,
        "flat": OutputLayout.FLAT,
        "single": OutputLayout.SINGLE,
    }
    return mapping[s]


def policy_from_string(s: str) -> DuplicatePolicy:
    """Convert string to DuplicatePolicy enum."""
    mapping = {
        "skip": DuplicatePolicy.SKIP,
        "keep-larger": DuplicatePolicy.KEEP_HIGHER_RESOLUTION,
        "keep-smaller": DuplicatePolicy.KEEP_FIRST,  # Fallback
        "keep-all": DuplicatePolicy.KEEP_BOTH,
    }
    return mapping[s]


def backend_from_string(s: str) -> HashBackend:
    """Convert string to HashBackend enum."""
    mapping = {
        "auto": HashBackend.AUTO,
        "cpu": HashBackend.CPU,
        "gpu-cuda": HashBackend.GPU_CUDA,
        "gpu-opencl": HashBackend.GPU_OPENCL,
    }
    return mapping[s]


def build_config(args: argparse.Namespace) -> SorterConfig:
    """Build SorterConfig from parsed arguments."""
    input_sources = parse_input_sources(args.input_dirs)
    
    db_path = args.db
    if db_path is None:
        db_path = args.output / ".gphotos-sorter.db"
    
    return SorterConfig(
        output_root=args.output,
        inputs=input_sources,
        layout=layout_from_string(args.layout),
        duplicate_policy=policy_from_string(args.duplicates),
        hash_backend=backend_from_string(args.hash_backend),
        workers=args.jobs or 4,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        db_path=db_path,
    )


def create_dependencies(
    config: SorterConfig,
    reporter: ProgressReporter,
) -> ProcessorDependencies:
    """Create all dependencies for the processor."""
    # Create hash engine
    hash_engine = create_hash_engine(config.hash_backend)
    
    # Create metadata extractor
    metadata_extractor = ExifToolMetadataExtractor()
    
    # Create repository
    assert config.db_path is not None
    repository = SQLiteMediaRepository(config.db_path)
    
    # Create services
    scanner = DirectoryScanner()
    deduplicator = DuplicateResolver(policy=config.duplicate_policy)
    file_manager = FileManager(
        output_root=config.output_root,
        layout=config.layout,
        dry_run=config.dry_run,
    )
    
    return ProcessorDependencies(
        hash_engine=hash_engine,
        metadata_extractor=metadata_extractor,
        repository=repository,
        scanner=scanner,
        deduplicator=deduplicator,
        file_manager=file_manager,
        progress=reporter,
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    # Create reporter based on verbosity settings
    if args.quiet:
        reporter = QuietProgressReporter()
    else:
        reporter = RichProgressReporter(verbose=args.verbose)
    
    try:
        # Build configuration
        config = build_config(args)
        
        # Print header and config
        reporter.print_header("Google Photos Sorter")
        reporter.print_config({
            "Input Sources": ", ".join(str(s.path) for s in config.inputs),
            "Output Directory": str(config.output_root),
            "Layout": config.layout.value,
            "Duplicate Policy": config.duplicate_policy.value,
            "Hash Backend": config.hash_backend.value,
            "Workers": config.workers,
            "Dry Run": config.dry_run,
        })
        
        # Create dependencies
        deps = create_dependencies(config, reporter)
        
        # Create and run processor
        processor = MediaProcessor(config=config, deps=deps)
        
        try:
            stats = processor.process()
            
            # Print final stats
            reporter.print_stats(stats)
            
            return 0 if stats.errors == 0 else 1
            
        except KeyboardInterrupt:
            reporter.warning("\nShutting down gracefully...")
            reporter.info("Flushing pending operations...")
            raise
            
        finally:
            # Cleanup
            try:
                deps.metadata_extractor.flush()
                deps.metadata_extractor.close()
            except Exception:
                pass
            try:
                deps.repository.close()
            except Exception:
                pass
    
    except KeyboardInterrupt:
        reporter.warning("Shutdown complete. Run again to resume.")
        return 130
    except Exception as e:
        reporter.error(f"Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
