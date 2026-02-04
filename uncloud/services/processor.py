"""Main media processor - orchestrates all services."""
from __future__ import annotations

import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ..core.config import SorterConfig
from ..core.models import (
    MediaItem, HashResult, ProcessingStats, CopyPlan,
)
from ..core.protocols import (
    HashEngine, MetadataExtractor, MediaRepository, ProgressReporter,
)
from .scanner import DirectoryScanner
from .deduplicator import DuplicateResolver
from .file_ops import FileManager


# Global interrupt flag
_interrupted = False


def _signal_handler(signum, frame):
    """Handle SIGINT gracefully."""
    global _interrupted
    _interrupted = True


@dataclass
class ProcessorDependencies:
    """All dependencies needed by the processor.
    
    This is explicitly passed in - no globals or singletons.
    """
    hash_engine: HashEngine
    metadata_extractor: MetadataExtractor
    repository: MediaRepository
    progress: ProgressReporter
    file_manager: FileManager
    scanner: DirectoryScanner
    deduplicator: DuplicateResolver


def _hash_item_worker(item_data: dict) -> dict:
    """Worker function for hashing a single item.
    
    Takes and returns dicts for pickling across processes.
    """
    import sys
    from pathlib import Path
    from ..engines.hash_engine import CPUHashEngine
    from ..engines.metadata import ExifToolMetadataExtractor
    
    # Suppress KeyboardInterrupt tracebacks in workers
    sys.tracebacklimit = 0
    
    path = Path(item_data["path"])
    
    engine = CPUHashEngine()
    extractor = ExifToolMetadataExtractor(use_batch_mode=False)
    
    try:
        # Compute hash
        similarity_hash = engine.compute_hash(path)
        
        # Extract metadata
        sidecar = Path(item_data["sidecar"]) if item_data.get("sidecar") else None
        date_taken, date_source = extractor.extract_datetime(path, sidecar)
        width, height = extractor.extract_resolution(path)
        
        return {
            **item_data,
            "similarity_hash": similarity_hash,
            "date_taken": date_taken.isoformat() if date_taken else None,
            "date_source": date_source,
            "width": width,
            "height": height,
            "error": None,
        }
    except Exception as e:
        return {
            **item_data,
            "error": str(e),
        }
    finally:
        extractor.close()


class MediaProcessor:
    """Main processor that orchestrates media processing.
    
    Uses two-phase processing:
    1. Hash Phase: Compute hashes for all files in parallel
    2. Copy Phase: Dedupe and copy files
    
    All dependencies are injected - no global state.
    """
    
    def __init__(self, config: SorterConfig, deps: ProcessorDependencies):
        """Initialize processor with config and dependencies.
        
        Args:
            config: Processing configuration.
            deps: All required dependencies.
        """
        self._config = config
        self._deps = deps
        self._stats = ProcessingStats()
        
        # Set up signal handler for graceful shutdown
        global _interrupted
        _interrupted = False
        signal.signal(signal.SIGINT, _signal_handler)
    
    def process(self) -> ProcessingStats:
        """Run the full processing pipeline.
        
        Returns:
            Statistics about what was processed.
        
        Raises:
            KeyboardInterrupt: If interrupted by user.
        """
        global _interrupted
        
        try:
            # Phase 0: Crash recovery
            self._recover_from_crash()
            
            # Phase 1: Scan and collect items
            items = self._scan_items()
            self._stats.total_files = len(items)
            
            if not items:
                self._deps.progress.info("No files to process")
                return self._stats
            
            # Apply limit if set
            if self._config.limit:
                items = items[:self._config.limit]
                self._stats.total_files = len(items)
            
            # Check for interrupt
            if _interrupted:
                raise KeyboardInterrupt()
            
            # Phase 2: Hash all items
            hash_results = self._hash_phase(items)
            
            # Check for interrupt
            if _interrupted:
                raise KeyboardInterrupt()
            
            # Phase 3: Dedupe and plan copies
            copy_plans = self._plan_copies(hash_results)
            
            # Check for interrupt
            if _interrupted:
                raise KeyboardInterrupt()
            
            # Phase 4: Execute copies
            if not self._config.dry_run:
                self._copy_phase(copy_plans)
            else:
                self._deps.progress.info(
                    f"DRY RUN: Would copy {len(copy_plans)} files"
                )
            
            return self._stats
        
        except KeyboardInterrupt:
            # Clean up any partial state
            self._deps.progress.warning(
                f"Interrupted. Processed {self._stats.copied + self._stats.skipped_duplicate} "
                f"of {self._stats.total_files} files."
            )
            raise
    
    def _recover_from_crash(self) -> None:
        """Clean up any pending operations from a previous crash."""
        pending = self._deps.repository.get_pending_operations()
        if not pending:
            return
        
        self._deps.progress.warning(
            f"Found {len(pending)} pending operations, cleaning up..."
        )
        
        for op in pending:
            if op.operation == "copy" and op.target_path:
                self._deps.file_manager.delete_file(Path(op.target_path))
        
        self._deps.repository.clear_all_pending_operations()
    
    def _scan_items(self) -> list[MediaItem]:
        """Scan input directories and collect items."""
        self._deps.progress.info("Scanning input directories...")
        
        # Get known paths to skip
        known_paths = self._deps.repository.get_all_source_paths()
        
        items = list(self._deps.scanner.scan(
            sources=list(self._config.inputs),
            recursive=self._config.recursive,
            skip_known=known_paths,
        ))
        
        self._deps.progress.info(
            f"Found {len(items)} files to process ({len(known_paths)} skipped)"
        )
        
        return items
    
    def _hash_phase(self, items: list[MediaItem]) -> list[HashResult]:
        """Phase 1: Hash all items using multiprocessing."""
        global _interrupted
        
        self._deps.progress.start_phase("Hashing", len(items))
        
        # Convert items to dicts for pickling
        item_dicts = [
            {
                "path": str(item.path),
                "owner": item.owner,
                "tags": list(item.tags),
                "sidecar": str(item.sidecar_path) if item.sidecar_path else None,
                "is_media": item.is_media,
            }
            for item in items
        ]
        
        results: list[HashResult] = []
        
        # Use ProcessPoolExecutor for true parallelism
        executor = ProcessPoolExecutor(max_workers=self._config.workers)
        try:
            futures = {
                executor.submit(_hash_item_worker, item_dict): i
                for i, item_dict in enumerate(item_dicts)
            }
            
            completed = 0
            for future in as_completed(futures):
                # Check for interrupt
                if _interrupted:
                    # Shutdown executor immediately
                    executor.shutdown(wait=False)
                    raise KeyboardInterrupt()
                
                try:
                    result_dict = future.result()
                    result = self._dict_to_hash_result(result_dict, items[futures[future]])
                    results.append(result)
                except Exception as e:
                    idx = futures[future]
                    results.append(HashResult(
                        item=items[idx],
                        error=str(e),
                    ))
                
                completed += 1
                self._deps.progress.update_phase(completed)
        finally:
            executor.shutdown(wait=False)
        
        self._deps.progress.info(f"Hashed {len(results)} files")
        self._deps.progress.end_phase()
        return results
    
    def _dict_to_hash_result(self, d: dict, item: MediaItem) -> HashResult:
        """Convert dict back to HashResult."""
        from datetime import datetime
        
        date_taken = None
        if d.get("date_taken"):
            try:
                date_taken = datetime.fromisoformat(d["date_taken"])
            except Exception:
                pass
        
        return HashResult(
            item=item,
            similarity_hash=d.get("similarity_hash"),
            date_taken=date_taken,
            date_source=d.get("date_source", "unknown"),
            width=d.get("width"),
            height=d.get("height"),
            error=d.get("error"),
        )
    
    def _plan_copies(self, hash_results: list[HashResult]) -> list[CopyPlan]:
        """Plan which files to copy based on deduplication."""
        self._deps.progress.info("Planning copies with deduplication...")
        
        # Group by hash
        groups = self._deps.deduplicator.group_by_hash(hash_results)
        
        plans: list[CopyPlan] = []
        
        for group in groups:
            # Check for interrupt
            if _interrupted:
                raise KeyboardInterrupt()
            
            # Check if hash exists in database
            existing = self._deps.repository.get_by_hash(group.hash_value)
            
            # Resolve duplicates
            best, duplicates = self._deps.deduplicator.resolve(group, existing)
            
            if best is None:
                # All duplicates of existing - skip
                for dup in group.items:
                    self._stats.skipped_duplicate += 1
                continue
            
            # Plan copy for best
            output_dir = self._deps.file_manager.build_output_directory(best.date_taken)
            target_path = self._deps.file_manager.find_unique_path(
                output_dir,
                best.date_taken,
                best.item.tags,
                best.item.extension,
                self._config.filename_format,
            )
            
            plans.append(CopyPlan(
                source=best.item,
                target_dir=output_dir,
                target_name=target_path.name,
                hash_result=best,
            ))
            
            # Count duplicates
            for _ in duplicates:
                self._stats.skipped_duplicate += 1
        
        self._deps.progress.info(f"Planned {len(plans)} copies")
        return plans
    
    def _copy_phase(self, plans: list[CopyPlan]) -> None:
        """Phase 2: Execute copy plans."""
        global _interrupted
        
        self._deps.progress.start_phase("Copying", len(plans))
        
        for i, plan in enumerate(plans):
            # Check for interrupt
            if _interrupted:
                raise KeyboardInterrupt()
            
            try:
                # Track pending operation for crash recovery
                op_id = self._deps.repository.add_pending_operation(
                    source=str(plan.source.path),
                    target=str(plan.target_path),
                    hash_val=plan.hash_result.similarity_hash or "",
                    op="copy",
                )
                
                # Copy file
                success = self._deps.file_manager.copy_file(
                    plan.source.path,
                    plan.target_path,
                )
                
                if success:
                    # Write EXIF if configured
                    if self._config.modify_exif and plan.hash_result.date_taken:
                        self._deps.metadata_extractor.write_tags(
                            plan.target_path,
                            {"DateTimeOriginal": plan.hash_result.date_taken.strftime("%Y:%m:%d %H:%M:%S")},
                        )
                    
                    # Update database
                    from ..persistence.database import MediaRecord
                    record = MediaRecord(
                        canonical_path=str(plan.target_path),
                        similarity_hash=plan.hash_result.similarity_hash,
                        owner=plan.source.owner,
                        date_taken=plan.hash_result.date_taken,
                        tags=",".join(plan.source.tags),
                        width=plan.hash_result.width,
                        height=plan.hash_result.height,
                        source_paths=str(plan.source.path),
                    )
                    self._deps.repository.upsert(record)
                    self._stats.copied += 1
                else:
                    self._stats.errors += 1
                
                # Complete pending operation
                self._deps.repository.complete_pending_operation(op_id)
                
            except Exception as e:
                self._deps.progress.error(f"Error copying {plan.source.path}: {e}")
                self._stats.errors += 1
            
            self._deps.progress.update_phase(i + 1)
        
        # Flush any pending EXIF writes
        self._deps.metadata_extractor.flush()
        
        self._deps.progress.info(
            f"Copied {self._stats.copied} files, {self._stats.errors} errors"
        )
        self._deps.progress.end_phase()
