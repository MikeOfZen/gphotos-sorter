"""Synchronized file operations - keeps FS and DB in sync.

This mid-layer ensures that any file operation (delete, rename, move)
updates both the filesystem AND the database atomically.

File is always the source of truth. Database is just an index.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from ..persistence.database import SQLiteMediaRepository, MediaRecord


@dataclass
class SyncResult:
    """Result of a synchronized operation."""
    success: bool
    message: str
    old_path: Optional[Path] = None
    new_path: Optional[Path] = None


class FileOpsSynchronizer:
    """Keeps filesystem and database in sync.
    
    All operations are atomic: either both FS and DB update, or neither.
    Uses pending operations table for crash recovery.
    """
    
    def __init__(
        self,
        repository: SQLiteMediaRepository,
        dry_run: bool = False,
    ):
        """Initialize synchronizer.
        
        Args:
            repository: Database repository.
            dry_run: If True, don't actually modify FS or DB.
        """
        self._repo = repository
        self._dry_run = dry_run
    
    def delete_file(self, path: Path) -> SyncResult:
        """Delete a file from both FS and DB.
        
        Args:
            path: File to delete.
            
        Returns:
            SyncResult with success status.
        """
        if not path.exists():
            return SyncResult(
                success=False,
                message=f"File not found: {path}",
                old_path=path,
            )
        
        if self._dry_run:
            return SyncResult(
                success=True,
                message=f"Would delete: {path}",
                old_path=path,
            )
        
        # Record pending operation for crash recovery
        op_id = self._repo.add_pending_operation(
            source=str(path),
            target="",
            hash_val="",
            op="delete",
        )
        
        try:
            # Delete from filesystem
            path.unlink()
            
            # Delete from database
            self._repo.delete_by_path(str(path))
            
            # Clear pending operation
            self._repo.complete_pending_operation(op_id)
            
            return SyncResult(
                success=True,
                message=f"Deleted: {path}",
                old_path=path,
            )
        except Exception as e:
            # Don't complete pending op - let recovery handle it
            return SyncResult(
                success=False,
                message=f"Error deleting {path}: {e}",
                old_path=path,
            )
    
    def rename_file(self, old_path: Path, new_path: Path) -> SyncResult:
        """Rename/move a file in both FS and DB.
        
        Args:
            old_path: Current file path.
            new_path: New file path.
            
        Returns:
            SyncResult with success status.
        """
        if not old_path.exists():
            return SyncResult(
                success=False,
                message=f"File not found: {old_path}",
                old_path=old_path,
            )
        
        if new_path.exists():
            return SyncResult(
                success=False,
                message=f"Target already exists: {new_path}",
                old_path=old_path,
                new_path=new_path,
            )
        
        if self._dry_run:
            return SyncResult(
                success=True,
                message=f"Would rename: {old_path} -> {new_path}",
                old_path=old_path,
                new_path=new_path,
            )
        
        # Record pending operation
        op_id = self._repo.add_pending_operation(
            source=str(old_path),
            target=str(new_path),
            hash_val="",
            op="rename",
        )
        
        try:
            # Ensure parent directory exists
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Rename in filesystem
            shutil.move(str(old_path), str(new_path))
            
            # Update in database
            self._repo.update_path(str(old_path), str(new_path))
            
            # Clear pending operation
            self._repo.complete_pending_operation(op_id)
            
            return SyncResult(
                success=True,
                message=f"Renamed: {old_path} -> {new_path}",
                old_path=old_path,
                new_path=new_path,
            )
        except Exception as e:
            # Attempt rollback if FS changed but DB didn't
            if new_path.exists() and not old_path.exists():
                try:
                    shutil.move(str(new_path), str(old_path))
                except Exception:
                    pass
            return SyncResult(
                success=False,
                message=f"Error renaming {old_path}: {e}",
                old_path=old_path,
                new_path=new_path,
            )
    
    def move_file(self, source: Path, target_dir: Path) -> SyncResult:
        """Move a file to a new directory in both FS and DB.
        
        Args:
            source: File to move.
            target_dir: Directory to move to.
            
        Returns:
            SyncResult with success status.
        """
        new_path = target_dir / source.name
        return self.rename_file(source, new_path)
    
    def copy_file(
        self,
        source: Path,
        target: Path,
        record: Optional[MediaRecord] = None,
    ) -> SyncResult:
        """Copy a file and add to DB.
        
        Args:
            source: Source file.
            target: Target path.
            record: Optional MediaRecord to insert.
            
        Returns:
            SyncResult with success status.
        """
        if not source.exists():
            return SyncResult(
                success=False,
                message=f"Source not found: {source}",
                old_path=source,
            )
        
        if target.exists():
            return SyncResult(
                success=False,
                message=f"Target already exists: {target}",
                old_path=source,
                new_path=target,
            )
        
        if self._dry_run:
            return SyncResult(
                success=True,
                message=f"Would copy: {source} -> {target}",
                old_path=source,
                new_path=target,
            )
        
        # Record pending operation
        hash_val = record.similarity_hash if record else ""
        op_id = self._repo.add_pending_operation(
            source=str(source),
            target=str(target),
            hash_val=hash_val or "",
            op="copy",
        )
        
        try:
            # Ensure parent directory exists
            target.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(source, target)
            
            # Add to database if record provided
            if record:
                record.canonical_path = str(target)
                self._repo.upsert(record)
            
            # Clear pending operation
            self._repo.complete_pending_operation(op_id)
            
            return SyncResult(
                success=True,
                message=f"Copied: {source} -> {target}",
                old_path=source,
                new_path=target,
            )
        except Exception as e:
            # Cleanup on failure
            if target.exists():
                try:
                    target.unlink()
                except Exception:
                    pass
            return SyncResult(
                success=False,
                message=f"Error copying {source}: {e}",
                old_path=source,
                new_path=target,
            )
    
    def delete_duplicates_by_hash(
        self,
        similarity_hash: str,
        keep_policy: str = "first",
    ) -> list[SyncResult]:
        """Delete duplicate files with the same hash.
        
        Args:
            similarity_hash: Hash to find duplicates for.
            keep_policy: Which file to keep: 'first', 'largest', 'smallest'
            
        Returns:
            List of SyncResults for each deletion.
        """
        records = self._repo.get_all_by_hash(similarity_hash)
        
        if len(records) <= 1:
            return []
        
        # Sort by policy
        if keep_policy == "largest":
            records.sort(key=lambda r: (r.width or 0) * (r.height or 0), reverse=True)
        elif keep_policy == "smallest":
            records.sort(key=lambda r: (r.width or 0) * (r.height or 0))
        # else: keep first (original order)
        
        # Keep first, delete rest
        to_keep = records[0]
        to_delete = records[1:]
        
        results = []
        for record in to_delete:
            path = Path(record.canonical_path)
            result = self.delete_file(path)
            results.append(result)
        
        return results
    
    def recover_pending_operations(self) -> list[SyncResult]:
        """Recover from any pending operations after a crash.
        
        Returns:
            List of recovery results.
        """
        pending = self._repo.get_pending_operations()
        results = []
        
        for op in pending:
            if op.operation == "delete":
                # File should be deleted
                path = Path(op.source_path)
                if path.exists():
                    result = self.delete_file(path)
                else:
                    # Already deleted, just clean up DB
                    self._repo.delete_by_path(op.source_path)
                    self._repo.complete_pending_operation(op.id)
                    result = SyncResult(
                        success=True,
                        message=f"Recovered delete: {path}",
                    )
                results.append(result)
            
            elif op.operation == "rename":
                old_path = Path(op.source_path)
                new_path = Path(op.target_path)
                
                if new_path.exists() and not old_path.exists():
                    # Rename completed on FS, update DB
                    self._repo.update_path(op.source_path, op.target_path)
                    self._repo.complete_pending_operation(op.id)
                    result = SyncResult(
                        success=True,
                        message=f"Recovered rename: {old_path} -> {new_path}",
                    )
                elif old_path.exists():
                    # Rename never happened, clear pending
                    self._repo.complete_pending_operation(op.id)
                    result = SyncResult(
                        success=True,
                        message=f"Cleared stale pending rename: {old_path}",
                    )
                else:
                    result = SyncResult(
                        success=False,
                        message=f"Cannot recover: both paths missing",
                    )
                results.append(result)
            
            elif op.operation == "copy":
                target = Path(op.target_path)
                if target.exists():
                    # Copy completed, ensure in DB
                    self._repo.complete_pending_operation(op.id)
                    result = SyncResult(
                        success=True,
                        message=f"Recovered copy: {target}",
                    )
                else:
                    # Copy failed, clean up
                    self._repo.complete_pending_operation(op.id)
                    result = SyncResult(
                        success=True,
                        message=f"Cleared failed copy: {target}",
                    )
                results.append(result)
        
        return results
