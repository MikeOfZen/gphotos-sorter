"""CLI with subcommands: import, index, dedupe, info, etc."""
from __future__ import annotations

import argparse
import signal
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
from .logging.rich_logger import RichProgressReporter, QuietProgressReporter


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="uncloud",
        description="Photo library management: import, organize, deduplicate.",
    )
    
    # Global options
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
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ============ IMPORT command ============
    import_parser = subparsers.add_parser(
        "import",
        help="Import and organize media files from source directories",
    )
    import_parser.add_argument(
        "input_dirs",
        nargs="+",
        type=Path,
        help="Input directories (format: PATH or owner:PATH)",
    )
    import_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory (your photo library)",
    )
    import_parser.add_argument(
        "--layout",
        type=str,
        choices=["year-month", "year-month-day", "flat", "single"],
        default="year-month",
        help="Output directory layout (default: year-month)",
    )
    import_parser.add_argument(
        "--owner",
        type=str,
        default=None,
        help="Owner/folder name (e.g., 'Mine', 'Family')",
    )
    import_parser.add_argument(
        "--duplicates",
        type=str,
        choices=["skip", "keep-larger", "keep-smaller", "keep-all"],
        default="skip",
        help="How to handle duplicates during import (default: skip)",
    )
    import_parser.add_argument(
        "-w", "-j", "--workers",
        dest="workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    import_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without copying",
    )
    import_parser.add_argument(
        "--hash-backend",
        type=str,
        choices=["auto", "cpu", "gpu-cuda", "gpu-opencl"],
        default="auto",
        help="Hash computation backend (default: auto)",
    )
    import_parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to database file (default: OUTPUT/.uncloud.db)",
    )
    import_parser.add_argument(
        "--write-hash",
        action="store_true",
        help="Write computed hash to file metadata for faster future reindexing",
    )
    
    # ============ INDEX command ============
    index_parser = subparsers.add_parser(
        "index",
        help="Build/rebuild database index from existing photo library",
    )
    index_parser.add_argument(
        "directory",
        type=Path,
        help="Directory to scan and index",
    )
    index_parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Database path (default: DIRECTORY/.uncloud.db)",
    )
    index_parser.add_argument(
        "-w", "--workers",
        type=int,
        default=32,
        help="Number of worker threads (default: 32)",
    )
    index_parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=64,
        help="Number of records to batch before writing to DB (default: 64, 0=disable)",
    )
    index_parser.add_argument(
        "--hash-backend",
        choices=["auto", "cpu", "gpu-cuda"],
        default="cpu",
        help="Hash backend (default: cpu)",
    )
    index_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without modifying database",
    )
    index_parser.add_argument(
        "--write-hash",
        action="store_true",
        help="Write computed hash to file metadata for faster future reindexing",
    )
    
    # ============ DEDUPE command ============
    dedupe_parser = subparsers.add_parser(
        "dedupe",
        help="Find and remove duplicate files from your library",
    )
    dedupe_parser.add_argument(
        "directory",
        type=Path,
        help="Directory to scan for duplicates",
    )
    dedupe_parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Database path (default: DIRECTORY/.uncloud.db)",
    )
    dedupe_parser.add_argument(
        "--policy",
        type=str,
        choices=["keep-first", "keep-largest", "keep-smallest"],
        default="keep-largest",
        help="Which duplicate to keep (default: keep-largest)",
    )
    dedupe_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show duplicates without deleting",
    )
    dedupe_parser.add_argument(
        "--min-duplicates",
        type=int,
        default=2,
        help="Minimum duplicates per hash to process (default: 2)",
    )
    
    # ============ INFO command ============
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about a file or the library database",
    )
    info_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="File or directory to get info about",
    )
    info_parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Database path",
    )
    info_parser.add_argument(
        "--show-duplicates",
        action="store_true",
        help="Show duplicate file groups",
    )
    
    # ============ DELETE command ============
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete a file from both filesystem and database",
    )
    delete_parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Files to delete",
    )
    delete_parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Database path",
    )
    delete_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )
    
    # ============ RENAME command ============
    rename_parser = subparsers.add_parser(
        "rename",
        help="Rename/move a file in both filesystem and database",
    )
    rename_parser.add_argument(
        "source",
        type=Path,
        help="Source file path",
    )
    rename_parser.add_argument(
        "dest",
        type=Path,
        help="Destination file path",
    )
    rename_parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Database path",
    )
    rename_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without renaming",
    )
    
    return parser


# ============ Command Handlers ============

def cmd_import(args: argparse.Namespace, reporter) -> int:
    """Handle the import command."""
    from .core.config import SorterConfig
    from .engines.hash_engine import create_hash_engine
    from .engines.metadata import ExifToolMetadataExtractor
    from .persistence.database import SQLiteMediaRepository
    from .services.scanner import DirectoryScanner
    from .services.deduplicator import DuplicateResolver
    from .services.file_ops import FileManager
    from .services.processor import MediaProcessor, ProcessorDependencies
    
    # Parse input sources
    sources = []
    for inp in args.input_dirs:
        inp_str = str(inp)
        if ":" in inp_str and not inp_str.startswith("/"):
            owner, path_str = inp_str.split(":", 1)
            sources.append(InputSource(path=Path(path_str), owner=owner))
        else:
            sources.append(InputSource(path=inp))
    
    # Build config
    db_path = args.db if args.db else args.output / ".uncloud.db"
    
    layout_map = {
        "year-month": OutputLayout.YEAR_MONTH,
        "year-month-day": OutputLayout.YEAR_MONTH_DAY,
        "flat": OutputLayout.FLAT,
        "single": OutputLayout.SINGLE,
    }
    
    policy_map = {
        "skip": DuplicatePolicy.SKIP,
        "keep-larger": DuplicatePolicy.KEEP_HIGHER_RESOLUTION,
        "keep-smaller": DuplicatePolicy.KEEP_FIRST,
        "keep-all": DuplicatePolicy.KEEP_BOTH,
    }
    
    backend_map = {
        "auto": HashBackend.AUTO,
        "cpu": HashBackend.CPU,
        "gpu-cuda": HashBackend.GPU_CUDA,
        "gpu-opencl": HashBackend.GPU_OPENCL,
    }
    
    config = SorterConfig(
        output_root=args.output,
        inputs=tuple(sources),
        layout=layout_map[args.layout],
        duplicate_policy=policy_map[args.duplicates],
        hash_backend=backend_map[args.hash_backend],
        workers=args.workers or 4,
        batch_size=100,
        dry_run=args.dry_run,
        db_path=db_path,
        owner_folder=args.owner,
    )
    
    # Print config
    reporter.print_header("uncloud import")
    reporter.print_config({
        "Input Sources": ", ".join(str(s.path) for s in config.inputs),
        "Output Directory": str(config.output_root),
        "Layout": config.layout.value,
        "Duplicate Policy": config.duplicate_policy.value,
        "Workers": config.workers,
        "Dry Run": config.dry_run,
    })
    
    # Create dependencies
    hash_engine = create_hash_engine(config.hash_backend)
    metadata_extractor = ExifToolMetadataExtractor()
    repository = SQLiteMediaRepository(config.db_path)
    scanner = DirectoryScanner()
    deduplicator = DuplicateResolver(policy=config.duplicate_policy)
    file_manager = FileManager(
        output_root=config.output_root,
        layout=config.layout,
        dry_run=config.dry_run,
        owner_folder=config.owner_folder,
    )
    
    deps = ProcessorDependencies(
        hash_engine=hash_engine,
        metadata_extractor=metadata_extractor,
        repository=repository,
        scanner=scanner,
        deduplicator=deduplicator,
        file_manager=file_manager,
        progress=reporter,
    )
    
    try:
        processor = MediaProcessor(config=config, deps=deps)
        stats = processor.process()
        reporter.print_stats(stats)
        return 0 if stats.errors == 0 else 1
    finally:
        metadata_extractor.close()
        repository.close()


def cmd_index(args: argparse.Namespace, reporter) -> int:
    """Handle the index command."""
    from .engines.hash_engine import create_hash_engine
    from .services.index_rebuilder import IndexRebuilder
    
    backend_map = {
        "auto": HashBackend.AUTO,
        "cpu": HashBackend.CPU,
        "gpu-cuda": HashBackend.GPU_CUDA,
    }
    
    output_dir = args.directory.resolve()
    db_path = args.db if args.db else output_dir / ".uncloud.db"
    
    hash_engine = create_hash_engine(backend_map[args.hash_backend])
    
    rebuilder = IndexRebuilder(
        hash_engine=hash_engine,
        progress=reporter,
        workers=args.workers,
        batch_size=args.batch_size,
        write_hash=args.write_hash,
    )
    
    stats = rebuilder.rebuild(
        output_dir=output_dir,
        db_path=db_path,
        dry_run=args.dry_run,
        backup=True,
    )
    
    return 0 if stats.errors < stats.total_files else 1


def cmd_dedupe(args: argparse.Namespace, reporter) -> int:
    """Handle the dedupe command."""
    from .persistence.database import SQLiteMediaRepository
    from .services.file_ops_sync import FileOpsSynchronizer
    
    directory = args.directory.resolve()
    db_path = args.db if args.db else directory / ".uncloud.db"
    
    if not db_path.exists():
        reporter.error(f"Database not found: {db_path}")
        reporter.info("Run 'uncloud index' first to build the database.")
        return 1
    
    reporter.print_header("uncloud dedupe")
    reporter.print_config({
        "Directory": str(directory),
        "Database": str(db_path),
        "Policy": args.policy,
        "Dry Run": args.dry_run,
    })
    
    repository = SQLiteMediaRepository(db_path)
    sync = FileOpsSynchronizer(repository=repository, dry_run=args.dry_run)
    
    try:
        # Get all duplicate hashes
        duplicates = repository.get_duplicate_hashes()
        
        if not duplicates:
            reporter.info("No duplicates found!")
            return 0
        
        total_dups = sum(count - 1 for _, count in duplicates)
        reporter.info(f"Found {len(duplicates)} hash groups with {total_dups} duplicate files")
        
        deleted_count = 0
        freed_bytes = 0
        
        for hash_val, count in duplicates:
            if count < args.min_duplicates:
                continue
            
            records = repository.get_all_by_hash(hash_val)
            
            # Sort by policy
            if args.policy == "keep-largest":
                records.sort(key=lambda r: (r.width or 0) * (r.height or 0), reverse=True)
            elif args.policy == "keep-smallest":
                records.sort(key=lambda r: (r.width or 0) * (r.height or 0))
            # else: keep-first (original order)
            
            # Keep first, delete rest
            to_keep = records[0]
            to_delete = records[1:]
            
            if args.verbose or args.dry_run:
                reporter.info(f"Hash {hash_val[:16]}... keeping: {Path(to_keep.canonical_path).name}")
            
            for record in to_delete:
                path = Path(record.canonical_path)
                
                if path.exists():
                    try:
                        freed_bytes += path.stat().st_size
                    except Exception:
                        pass
                
                result = sync.delete_file(path)
                
                if result.success:
                    deleted_count += 1
                    if args.verbose:
                        reporter.info(f"  Deleted: {path.name}")
                else:
                    reporter.warning(f"  Failed: {result.message}")
        
        # Summary
        reporter.print_header("Dedupe Complete")
        freed_mb = freed_bytes / (1024 * 1024)
        reporter.info(f"Deleted {deleted_count} duplicate files")
        reporter.info(f"Freed {freed_mb:.1f} MB")
        
        return 0
        
    finally:
        repository.close()


def cmd_info(args: argparse.Namespace, reporter) -> int:
    """Handle the info command."""
    from .persistence.database import SQLiteMediaRepository
    from .engines.metadata import ExifToolMetadataExtractor
    
    if args.path and args.path.is_file():
        # Show info about a single file
        extractor = ExifToolMetadataExtractor(use_batch_mode=False)
        try:
            reporter.print_header(f"File: {args.path.name}")
            
            # Extract metadata
            meta = extractor.extract_uncloud_metadata(args.path)
            dt, dt_src = extractor.extract_datetime(args.path, None)
            w, h = extractor.extract_resolution(args.path)
            
            reporter.info(f"Path: {args.path}")
            reporter.info(f"Size: {args.path.stat().st_size / 1024:.1f} KB")
            reporter.info(f"Resolution: {w}x{h}" if w and h else "Resolution: unknown")
            reporter.info(f"Date Taken: {dt} ({dt_src})" if dt else "Date Taken: unknown")
            reporter.info(f"Stored Hash: {meta['hash'] or 'none'}")
            reporter.info(f"Tags: {', '.join(meta['tags']) if meta['tags'] else 'none'}")
            
            return 0
        finally:
            extractor.close()
    
    # Show database info
    db_path = args.db or (args.path / ".uncloud.db" if args.path else None)
    
    if not db_path or not db_path.exists():
        reporter.error("Database not found. Specify --db or provide a directory path.")
        return 1
    
    repository = SQLiteMediaRepository(db_path)
    try:
        total = repository.count_all()
        duplicates = repository.get_duplicate_hashes()
        total_dups = sum(count - 1 for _, count in duplicates)
        
        reporter.print_header("Library Info")
        reporter.info(f"Database: {db_path}")
        reporter.info(f"Total indexed files: {total}")
        reporter.info(f"Unique hashes: {total - total_dups}")
        reporter.info(f"Duplicate files: {total_dups}")
        reporter.info(f"Duplicate groups: {len(duplicates)}")
        
        if args.show_duplicates and duplicates:
            reporter.info("\nDuplicate groups:")
            for hash_val, count in duplicates[:20]:
                reporter.info(f"  {hash_val[:16]}... ({count} files)")
            if len(duplicates) > 20:
                reporter.info(f"  ... and {len(duplicates) - 20} more groups")
        
        return 0
    finally:
        repository.close()


def cmd_delete(args: argparse.Namespace, reporter) -> int:
    """Handle the delete command."""
    from .persistence.database import SQLiteMediaRepository
    from .services.file_ops_sync import FileOpsSynchronizer
    
    repository = SQLiteMediaRepository(args.db)
    sync = FileOpsSynchronizer(repository=repository, dry_run=args.dry_run)
    
    try:
        deleted = 0
        for path in args.files:
            result = sync.delete_file(path.resolve())
            if result.success:
                reporter.info(result.message)
                deleted += 1
            else:
                reporter.error(result.message)
        
        reporter.info(f"Deleted {deleted}/{len(args.files)} files")
        return 0 if deleted == len(args.files) else 1
    finally:
        repository.close()


def cmd_rename(args: argparse.Namespace, reporter) -> int:
    """Handle the rename command."""
    from .persistence.database import SQLiteMediaRepository
    from .services.file_ops_sync import FileOpsSynchronizer
    
    repository = SQLiteMediaRepository(args.db)
    sync = FileOpsSynchronizer(repository=repository, dry_run=args.dry_run)
    
    try:
        result = sync.rename_file(args.source.resolve(), args.dest.resolve())
        if result.success:
            reporter.info(result.message)
            return 0
        else:
            reporter.error(result.message)
            return 1
    finally:
        repository.close()


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Create reporter
    if getattr(args, 'quiet', False):
        reporter = QuietProgressReporter()
    else:
        reporter = RichProgressReporter(verbose=getattr(args, 'verbose', False))
    
    # No command specified - show help
    if not args.command:
        parser.print_help()
        return 0
    
    # Dispatch to command handler
    try:
        if args.command == "import":
            return cmd_import(args, reporter)
        elif args.command == "index":
            return cmd_index(args, reporter)
        elif args.command == "dedupe":
            return cmd_dedupe(args, reporter)
        elif args.command == "info":
            return cmd_info(args, reporter)
        elif args.command == "delete":
            return cmd_delete(args, reporter)
        elif args.command == "rename":
            return cmd_rename(args, reporter)
        else:
            reporter.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C - no stack trace
        return 130
    except Exception as e:
        reporter.error(f"Error: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
