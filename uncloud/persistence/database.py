"""SQLite-based media repository."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class MediaRecord:
    """A record in the media database."""
    canonical_path: str
    similarity_hash: Optional[str] = None
    owner: str = ""
    date_taken: Optional[datetime] = None
    tags: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    source_paths: str = ""  # JSON list
    created_at: Optional[datetime] = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = field(default_factory=datetime.now)
    id: Optional[int] = None


@dataclass
class PendingOperation:
    """A pending file operation for crash recovery."""
    id: int
    source_path: str
    target_path: str
    similarity_hash: str
    operation: str
    created_at: datetime


class SQLiteMediaRepository:
    """SQLite implementation of the media repository.
    
    Handles all database operations for media records.
    """
    
    def __init__(self, db_path: Path):
        """Initialize repository with database path.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_database()
    
    def _init_database(self) -> None:
        """Create tables if they don't exist."""
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        
        cursor = self._conn.cursor()
        
        # Main media table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_path TEXT UNIQUE NOT NULL,
                similarity_hash TEXT,
                owner TEXT,
                date_taken TEXT,
                tags TEXT,
                width INTEGER,
                height INTEGER,
                source_paths TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index for hash lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_media_hash 
            ON media(similarity_hash)
        """)
        
        # Source path index for O(1) lookups
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS source_path_index (
                path TEXT PRIMARY KEY,
                media_hash TEXT
            )
        """)
        
        # Pending operations for crash recovery
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT NOT NULL,
                target_path TEXT,
                similarity_hash TEXT,
                operation TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self._conn.commit()
    
    def get_by_hash(self, similarity_hash: str) -> Optional[MediaRecord]:
        """Get record by similarity hash."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM media WHERE similarity_hash = ?",
            (similarity_hash,)
        )
        row = cursor.fetchone()
        return self._row_to_record(row) if row else None
    
    def has_source_path(self, path: str) -> bool:
        """Check if source path already processed."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT 1 FROM source_path_index WHERE path = ?",
            (path,)
        )
        return cursor.fetchone() is not None
    
    def get_all_source_paths(self) -> set[str]:
        """Get all known source paths."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT path FROM source_path_index")
        return {row[0] for row in cursor.fetchall()}
    
    def upsert(self, record: MediaRecord) -> None:
        """Insert or update a record."""
        cursor = self._conn.cursor()
        
        cursor.execute("""
            INSERT INTO media (
                canonical_path, similarity_hash, owner, date_taken,
                tags, width, height, source_paths, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(canonical_path) DO UPDATE SET
                similarity_hash = excluded.similarity_hash,
                owner = excluded.owner,
                date_taken = excluded.date_taken,
                tags = excluded.tags,
                width = excluded.width,
                height = excluded.height,
                source_paths = excluded.source_paths,
                updated_at = CURRENT_TIMESTAMP
        """, (
            record.canonical_path,
            record.similarity_hash,
            record.owner,
            record.date_taken.isoformat() if record.date_taken else None,
            record.tags,
            record.width,
            record.height,
            record.source_paths,
        ))
        
        # Update source path index
        if record.source_paths:
            for path in record.source_paths.split(","):
                path = path.strip()
                if path:
                    cursor.execute("""
                        INSERT OR REPLACE INTO source_path_index (path, media_hash)
                        VALUES (?, ?)
                    """, (path, record.similarity_hash))
        
        self._conn.commit()
    
    def add_pending_operation(
        self, 
        source: str, 
        target: str, 
        hash_val: str, 
        op: str
    ) -> int:
        """Track a pending operation for crash recovery."""
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO pending_operations (source_path, target_path, similarity_hash, operation)
            VALUES (?, ?, ?, ?)
        """, (source, target, hash_val, op))
        self._conn.commit()
        return cursor.lastrowid
    
    def complete_pending_operation(self, op_id: int) -> None:
        """Mark operation as complete."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM pending_operations WHERE id = ?", (op_id,))
        self._conn.commit()
    
    def get_pending_operations(self) -> list[PendingOperation]:
        """Get all pending operations."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM pending_operations")
        return [
            PendingOperation(
                id=row["id"],
                source_path=row["source_path"],
                target_path=row["target_path"] or "",
                similarity_hash=row["similarity_hash"] or "",
                operation=row["operation"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in cursor.fetchall()
        ]
    
    def clear_all_pending_operations(self) -> int:
        """Clear all pending operations."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM pending_operations")
        count = cursor.rowcount
        self._conn.commit()
        return count
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def _row_to_record(self, row: sqlite3.Row) -> MediaRecord:
        """Convert database row to MediaRecord."""
        date_taken = None
        if row["date_taken"]:
            try:
                date_taken = datetime.fromisoformat(row["date_taken"])
            except Exception:
                pass
        
        return MediaRecord(
            id=row["id"],
            canonical_path=row["canonical_path"],
            similarity_hash=row["similarity_hash"],
            owner=row["owner"] or "",
            date_taken=date_taken,
            tags=row["tags"] or "",
            width=row["width"],
            height=row["height"],
            source_paths=row["source_paths"] or "",
        )
    
    def __enter__(self) -> "SQLiteMediaRepository":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
