"""Index rebuilder service - rebuilds database index from filesystem."""
from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.protocols import HashEngine, ProgressReporter
from ..persistence.database import SQLiteMediaRepository, MediaRecord
from ..engines.metadata import ThreadLocalExifToolDaemon


# Supported media extensions
MEDIA_EXTENSIONS = frozenset({
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic', '.heif',
    '.mp4', '.mov', '.avi', '.mkv', '.m4v', '.3gp', '.wmv'
})


@dataclass
class RebuildStats:
    """Statistics from an index rebuild operation."""
    total_files: int = 0
    inserted: int = 0
    skipped_duplicates: int = 0
    errors: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 100.0
        return (self.inserted + self.skipped_duplicates) / self.total_files * 100


class IndexRebuilder:
    """Service for rebuilding the database index from filesystem.
    
    Scans an output directory for media files and rebuilds the hash index
    without re-copying files. Useful for:
    - Recovering from database corruption
    - Syncing database after manual file operations
    - Initial indexing of an existing photo library
    
    Follows the same dependency injection pattern as other services.
    """
    
    def __init__(
        self,
        hash_engine: HashEngine,
        progress: ProgressReporter,
        workers: int = 4,
        batch_size: int = 64,
    ):
        """Initialize the rebuilder.
        
        Args:
            hash_engine: Engine for computing file hashes.
            progress: Reporter for progress updates.
            workers: Number of parallel workers for hashing.
            batch_size: Number of records to batch before DB write (0=disable).
        """
        self._hash_engine = hash_engine
        self._progress = progress
        self._batch_size = batch_size
        self._workers = workers
    
    def scan_media_files(self, directory: Path) -> list[Path]:
        """Scan directory for media files.
        
        Args:
            directory: Root directory to scan.
            
        Returns:
            List of paths to media files found.
        """
        self._progress.info("Scanning for media files...")
        
        media_files = []
        for file in directory.rglob("*"):
            if file.is_file() and file.suffix.lower() in MEDIA_EXTENSIONS:
                media_files.append(file)
        
        self._progress.info(f"Found {len(media_files)} media files")
        return media_files
    
    def rebuild(
        self,
        output_dir: Path,
        db_path: Path,
        dry_run: bool = False,
        backup: bool = True,
    ) -> RebuildStats:
        """Rebuild the database index from filesystem.
        
        Args:
            output_dir: Directory containing organized media files.
            db_path: Path to the database file.
            dry_run: If True, scan but don't modify database.
            backup: If True, backup existing database before rebuild.
            
        Returns:
            Statistics about the rebuild operation.
        """
        stats = RebuildStats()
        
        # Validate output directory
        if not output_dir.exists():
            self._progress.error(f"Output directory not found: {output_dir}")
            return stats
        
        self._progress.print_header("Rebuild Database Index")
        self._progress.print_config({
            "Output Directory": str(output_dir),
            "Database": str(db_path),
            "Workers": self._workers,
            "Dry Run": dry_run,
        })
        
        # Scan for files
        media_files = self.scan_media_files(output_dir)
        stats.total_files = len(media_files)
        
        if not media_files:
            self._progress.warning("No media files found!")
            return stats
        
        # Dry run - just show what would be done
        if dry_run:
            self._print_dry_run_summary(media_files, output_dir)
            return stats
        
        # Backup existing database
        if backup and db_path.exists():
            self._backup_database(db_path)
        
        # Delete and recreate database
        if db_path.exists():
            self._progress.info("Deleting old database...")
            db_path.unlink()
        
        self._progress.info("Creating new database...")
        repo = SQLiteMediaRepository(db_path)
        
        try:
            # Process files
            stats = self._process_files(
                media_files=media_files,
                output_dir=output_dir,
                repository=repo,
            )
            
            # Print summary
            self._print_summary(stats, db_path)
            
        finally:
            repo.close()
        
        return stats
    
    def _backup_database(self, db_path: Path) -> Path:
        """Create a backup of the existing database.
        
        Args:
            db_path: Path to the database file.
            
        Returns:
            Path to the backup file.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}.db"
        self._progress.info(f"Backing up database to: {backup_path}")
        shutil.copy2(db_path, backup_path)
        return backup_path
    
    def _process_files(
        self,
        media_files: list[Path],
        output_dir: Path,
        repository: SQLiteMediaRepository,
    ) -> RebuildStats:
        """Process all media files and insert into database.
        
        Args:
            media_files: List of media file paths.
            output_dir: Root output directory (for relative path calculation).
            repository: Database repository to insert records.
            
        Returns:
            Statistics about the operation.
        """
        stats = RebuildStats(total_files=len(media_files))
        duplicates: dict[str, list[Path]] = {}
        first_errors: list[tuple[Path, str | None]] = []
        from_metadata_count = 0
        computed_count = 0
        
        # Thread-local ExifTool daemons (one per worker thread)
        exiftool_daemons = ThreadLocalExifToolDaemon()
        
        def hash_file(file_path: Path) -> tuple[Path, str | None, str | None, bool]:
            """Hash a single file. Returns (path, hash, error_msg, from_metadata).
            
            First checks if hash is stored in file metadata (fast).
            If not found, computes hash (slow).
            Uses thread-local ExifTool daemon for efficiency.
            """
            try:
                # Try to read hash from file metadata first (using daemon)
                daemon = exiftool_daemons.get_daemon()
                stored_hash = daemon.extract_hash(file_path)
                if stored_hash:
                    return file_path, stored_hash, None, True
                
                # Not in metadata, compute it
                computed_hash = self._hash_engine.compute_hash(file_path)
                return file_path, computed_hash, None, False
            except Exception as e:
                return file_path, None, str(e), False
        
        # Start progress tracking
        self._progress.start_phase("Indexing", stats.total_files)
        
        completed = 0
        batch: list[MediaRecord] = []
        
        def flush_batch():
            """Flush accumulated batch to database."""
            if not batch:
                return
            repository.batch_upsert(batch)
            batch.clear()
        
        executor = ThreadPoolExecutor(max_workers=self._workers)
        try:
            futures = {executor.submit(hash_file, fp): fp for fp in media_files}
            pending = set(futures.keys())
            
            while pending:
                # Wait for completions with timeout for interrupt checking
                import concurrent.futures
                done, pending = concurrent.futures.wait(
                    pending, timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    try:
                        file_path, hash_value, err_msg, from_meta = future.result(timeout=0)
                        completed += 1
                        
                        if from_meta:
                            from_metadata_count += 1
                        elif hash_value:
                            computed_count += 1
                        
                        # Update progress bar with total completed
                        self._progress.update_phase(completed)
                        
                        # Handle errors
                        if not hash_value:
                            stats.errors += 1
                            if len(first_errors) < 5:
                                first_errors.append((file_path, err_msg))
                            continue
                        
                        # Track duplicates for reporting (but still insert all)
                        if hash_value in duplicates:
                            duplicates[hash_value].append(file_path)
                            stats.skipped_duplicates += 1  # Count as duplicate but still insert
                        else:
                            duplicates[hash_value] = [file_path]
                        
                        # Extract owner from path structure
                        owner = self._extract_owner(file_path, output_dir)
                        
                        # Create record - DB indexes ALL files on filesystem
                        # (duplicates have same hash but different canonical_path)
                        record = MediaRecord(
                            canonical_path=str(file_path),
                            similarity_hash=hash_value,
                            owner=owner,
                            date_taken=None,
                            tags="",
                            width=None,
                            height=None,
                            source_paths=str(file_path),
                        )
                        
                        # Add to batch
                        if self._batch_size > 0:
                            batch.append(record)
                            stats.inserted += 1
                            # Flush when batch is full
                            if len(batch) >= self._batch_size:
                                flush_batch()
                        else:
                            # Batching disabled, insert immediately
                            repository.upsert(record)
                            stats.inserted += 1
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        self._progress.warning(f"Error: {e}")
                        stats.errors += 1
            
            # Flush any remaining records in batch
            flush_batch()
        except KeyboardInterrupt:
            # Flush what we have so far
            flush_batch()
            # Cancel all pending futures
            for f in pending:
                f.cancel()
            self._progress.end_phase()
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            # Clean up all ExifTool daemon processes
            exiftool_daemons.close_all()
        
        # End progress tracking
        self._progress.end_phase()
        
        # Show summary
        self._progress.info(
            f"Indexed {stats.inserted} files "
            f"({from_metadata_count} from metadata, {computed_count} computed)"
        )
        
        # Report errors
        if first_errors:
            self._progress.warning("First few errors:")
            for fp, msg in first_errors:
                self._progress.warning(f"  {fp.name}: {msg}")
        
        # Report duplicate groups
        self._report_duplicates(duplicates, output_dir)
        
        return stats
    
    def _extract_owner(self, file_path: Path, output_dir: Path) -> str:
        """Extract owner from file path structure.
        
        Assumes structure: output_dir/Owner/YYYY/MM/file.jpg
        
        Args:
            file_path: Path to the media file.
            output_dir: Root output directory.
            
        Returns:
            Owner name or "unknown".
        """
        try:
            parts = file_path.relative_to(output_dir).parts
            return parts[0] if len(parts) > 0 else "unknown"
        except ValueError:
            return "unknown"
    
    def _print_dry_run_summary(
        self,
        media_files: list[Path],
        output_dir: Path,
    ) -> None:
        """Print summary for dry run mode."""
        self._progress.info("Sample files that would be indexed:")
        for f in media_files[:10]:
            self._progress.info(f"  {f.relative_to(output_dir)}")
        if len(media_files) > 10:
            self._progress.info(f"  ... and {len(media_files) - 10} more")
        self._progress.info("")
        self._progress.info("To rebuild the index, run without --dry-run")
    
    def _print_summary(self, stats: RebuildStats, db_path: Path) -> None:
        """Print final summary of rebuild operation."""
        from ..core.models import ProcessingStats
        
        self._progress.print_header("Rebuild Complete")
        
        # Create ProcessingStats object for the reporter
        proc_stats = ProcessingStats(
            total_files=stats.total_files,
            copied=stats.inserted,
            skipped_duplicate=stats.skipped_duplicates,
            errors=stats.errors,
        )
        self._progress.print_stats(proc_stats)
        self._progress.info(f"Database saved to: {db_path}")
    
    def _report_duplicates(
        self,
        duplicates: dict[str, list[Path]],
        output_dir: Path,
    ) -> None:
        """Report duplicate file groups."""
        dup_hashes = [h for h, files in duplicates.items() if len(files) > 1]
        
        if not dup_hashes:
            return
        
        self._progress.warning(
            f"Found {len(dup_hashes)} hash groups with duplicate files"
        )
        
        for hash_val in dup_hashes[:10]:
            files = duplicates[hash_val]
            self._progress.info(f"  Hash {hash_val[:16]}... ({len(files)} files):")
            for f in files:
                try:
                    self._progress.info(f"    {f.relative_to(output_dir)}")
                except ValueError:
                    self._progress.info(f"    {f}")
        
        if len(dup_hashes) > 10:
            self._progress.info(f"  ... and {len(dup_hashes) - 10} more groups")
