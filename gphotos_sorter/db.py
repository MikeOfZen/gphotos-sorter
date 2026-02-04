from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class MediaRecord:
    similarity_hash: str
    canonical_path: str
    owner: str
    date_taken: Optional[str]
    date_source: str
    tags: list[str]
    source_paths: list[str]
    status: str
    notes: Optional[str]
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class PendingOperation:
    """A pending file operation for crash recovery."""
    source_path: str
    target_path: str
    similarity_hash: str
    operation: str  # "copy" or "delete"


class MediaDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.connection = sqlite3.connect(str(db_path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def close(self) -> None:
        self.connection.close()

    def _ensure_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS media (
                id INTEGER PRIMARY KEY,
                similarity_hash TEXT UNIQUE,
                canonical_path TEXT,
                owner TEXT,
                date_taken TEXT,
                date_source TEXT,
                tags TEXT,
                source_paths TEXT,
                status TEXT,
                notes TEXT,
                width INTEGER,
                height INTEGER
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_media_hash ON media(similarity_hash)"
        )
        # Source path index for O(1) lookups
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS source_path_index (
                path TEXT PRIMARY KEY,
                media_hash TEXT NOT NULL
            )
            """
        )
        # Pending operations for crash recovery
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_operations (
                id INTEGER PRIMARY KEY,
                source_path TEXT NOT NULL,
                target_path TEXT NOT NULL,
                similarity_hash TEXT NOT NULL,
                operation TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.commit()
        # Migrate existing source_paths to index if needed
        self._migrate_source_paths_to_index()

    def _migrate_source_paths_to_index(self) -> None:
        """Migrate existing source_paths from JSON to index table (one-time)."""
        # Check if migration needed by counting rows in index vs media
        cursor = self.connection.execute("SELECT COUNT(*) FROM source_path_index")
        index_count = cursor.fetchone()[0]
        cursor = self.connection.execute("SELECT COUNT(*) FROM media")
        media_count = cursor.fetchone()[0]
        
        if index_count > 0 or media_count == 0:
            return  # Already migrated or empty DB
        
        # Migrate all existing source_paths
        cursor = self.connection.execute("SELECT similarity_hash, source_paths FROM media")
        for row in cursor:
            sim_hash, paths_json = row
            if paths_json:
                paths = json.loads(paths_json)
                for path in paths:
                    normalized = os.path.normpath(path)
                    try:
                        self.connection.execute(
                            "INSERT OR IGNORE INTO source_path_index (path, media_hash) VALUES (?, ?)",
                            (normalized, sim_hash),
                        )
                    except sqlite3.IntegrityError:
                        pass  # Path already exists
        self.connection.commit()

    def get_by_hash(self, similarity_hash: str) -> Optional[MediaRecord]:
        cursor = self.connection.execute(
            "SELECT similarity_hash, canonical_path, owner, date_taken, date_source, tags, source_paths, status, notes, width, height FROM media WHERE similarity_hash = ?",
            (similarity_hash,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        tags = json.loads(row[5]) if row[5] else []
        source_paths = json.loads(row[6]) if row[6] else []
        return MediaRecord(
            similarity_hash=row[0],
            canonical_path=row[1],
            owner=row[2],
            date_taken=row[3],
            date_source=row[4],
            tags=tags,
            source_paths=source_paths,
            status=row[7],
            notes=row[8],
            width=row[9],
            height=row[10],
        )

    def upsert(self, record: MediaRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO media (
                similarity_hash, canonical_path, owner, date_taken, date_source, tags, source_paths, status, notes, width, height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(similarity_hash) DO UPDATE SET
                canonical_path = excluded.canonical_path,
                owner = excluded.owner,
                date_taken = excluded.date_taken,
                date_source = excluded.date_source,
                tags = excluded.tags,
                source_paths = excluded.source_paths,
                status = excluded.status,
                notes = excluded.notes,
                width = excluded.width,
                height = excluded.height
            """,
            (
                record.similarity_hash,
                record.canonical_path,
                record.owner,
                record.date_taken,
                record.date_source,
                json.dumps(sorted(set(record.tags))),
                json.dumps(sorted(set(record.source_paths))),
                record.status,
                record.notes,
                record.width,
                record.height,
            ),
        )
        # Update source_path_index
        for path in record.source_paths:
            normalized = os.path.normpath(path)
            self.connection.execute(
                "INSERT OR REPLACE INTO source_path_index (path, media_hash) VALUES (?, ?)",
                (normalized, record.similarity_hash),
            )
        self.connection.commit()

    def update_existing(self, similarity_hash: str, tags: Iterable[str], source_paths: Iterable[str]) -> None:
        existing = self.get_by_hash(similarity_hash)
        if not existing:
            return
        source_paths_list = list(source_paths)
        merged_tags = sorted(set(existing.tags).union(tags))
        merged_sources = sorted(set(existing.source_paths).union(source_paths_list))
        self.connection.execute(
            "UPDATE media SET tags = ?, source_paths = ? WHERE similarity_hash = ?",
            (json.dumps(merged_tags), json.dumps(merged_sources), similarity_hash),
        )
        # Update source_path_index for new paths
        for path in source_paths_list:
            normalized = os.path.normpath(path)
            self.connection.execute(
                "INSERT OR REPLACE INTO source_path_index (path, media_hash) VALUES (?, ?)",
                (normalized, similarity_hash),
            )
        self.connection.commit()

    def has_source_path(self, source_path: str) -> bool:
        """Check if a source path is already recorded (O(1) lookup via index)."""
        normalized = os.path.normpath(source_path)
        cursor = self.connection.execute(
            "SELECT 1 FROM source_path_index WHERE path = ? LIMIT 1",
            (normalized,),
        )
        return cursor.fetchone() is not None

    def get_all_source_paths(self) -> set[str]:
        """Get all source paths from index table for fast lookup."""
        cursor = self.connection.execute("SELECT path FROM source_path_index")
        return {row[0] for row in cursor}

    # Pending operations for crash recovery
    def add_pending_operation(self, source_path: str, target_path: str, similarity_hash: str, operation: str) -> int:
        """Record a pending operation before executing it."""
        cursor = self.connection.execute(
            "INSERT INTO pending_operations (source_path, target_path, similarity_hash, operation) VALUES (?, ?, ?, ?)",
            (source_path, target_path, similarity_hash, operation),
        )
        self.connection.commit()
        return cursor.lastrowid
    
    def complete_pending_operation(self, op_id: int) -> None:
        """Remove a pending operation after successful completion."""
        self.connection.execute("DELETE FROM pending_operations WHERE id = ?", (op_id,))
        self.connection.commit()
    
    def get_pending_operations(self) -> list[PendingOperation]:
        """Get all pending operations from a previous crashed run."""
        cursor = self.connection.execute(
            "SELECT source_path, target_path, similarity_hash, operation FROM pending_operations"
        )
        return [
            PendingOperation(
                source_path=row[0],
                target_path=row[1],
                similarity_hash=row[2],
                operation=row[3],
            )
            for row in cursor
        ]
    
    def clear_all_pending_operations(self) -> int:
        """Clear all pending operations (after recovery)."""
        cursor = self.connection.execute("SELECT COUNT(*) FROM pending_operations")
        count = cursor.fetchone()[0]
        self.connection.execute("DELETE FROM pending_operations")
        self.connection.commit()
        return count
