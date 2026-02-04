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
    source_paths: str = ""  # JSON list of source paths
    faces_hashes: str = ""  # JSON list of face recognition hashes (future)
    ai_desc: str = ""  # AI-generated description (future)
    objects: str = ""  # JSON list of detected objects (future)
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
                faces_hashes TEXT,
                ai_desc TEXT,
                objects TEXT,
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
        
        # Run migrations for existing databases
        self._migrate_schema()
    
    def _migrate_schema(self) -> None:
        """Apply schema migrations to existing databases."""
        assert self._conn is not None
        cursor = self._conn.cursor()
        
        # Check if we need to add new columns
        cursor.execute("PRAGMA table_info(media)")
        columns = {row[1] for row in cursor.fetchall()}
        
        # Add faces_hashes column if missing
        if "faces_hashes" not in columns:
            cursor.execute("ALTER TABLE media ADD COLUMN faces_hashes TEXT")
        
        # Add ai_desc column if missing
        if "ai_desc" not in columns:
            cursor.execute("ALTER TABLE media ADD COLUMN ai_desc TEXT")
        
        # Add objects column if missing
        if "objects" not in columns:
            cursor.execute("ALTER TABLE media ADD COLUMN objects TEXT")
        
        self._conn.commit()
    
    def get_by_hash(self, similarity_hash: str) -> Optional[MediaRecord]:
        """Get record by similarity hash."""
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM media WHERE similarity_hash = ?",
            (similarity_hash,)
        )
        row = cursor.fetchone()
        return self._row_to_record(row) if row else None
    
    def has_source_path(self, path: str) -> bool:
        """Check if source path already processed."""
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT 1 FROM source_path_index WHERE path = ?",
            (path,)
        )
        return cursor.fetchone() is not None
    
    def get_all_source_paths(self) -> set[str]:
        """Get all known source paths."""
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute("SELECT path FROM source_path_index")
        return {row[0] for row in cursor.fetchall()}
    
    def upsert(self, record: MediaRecord) -> None:
        """Insert or update a record."""
        assert self._conn is not None
        cursor = self._conn.cursor()
        
        cursor.execute("""
            INSERT INTO media (
                canonical_path, similarity_hash, owner, date_taken,
                tags, width, height, source_paths, 
                faces_hashes, ai_desc, objects, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(canonical_path) DO UPDATE SET
                similarity_hash = excluded.similarity_hash,
                owner = excluded.owner,
                date_taken = excluded.date_taken,
                tags = excluded.tags,
                width = excluded.width,
                height = excluded.height,
                source_paths = excluded.source_paths,
                faces_hashes = excluded.faces_hashes,
                ai_desc = excluded.ai_desc,
                objects = excluded.objects,
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
            record.faces_hashes,
            record.ai_desc,
            record.objects,
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
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO pending_operations (source_path, target_path, similarity_hash, operation)
            VALUES (?, ?, ?, ?)
        """, (source, target, hash_val, op))
        self._conn.commit()
        return cursor.lastrowid or 0
    
    def complete_pending_operation(self, op_id: int) -> None:
        """Mark operation as complete."""
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM pending_operations WHERE id = ?", (op_id,))
        self._conn.commit()
    
    def get_pending_operations(self) -> list[PendingOperation]:
        """Get all pending operations."""
        assert self._conn is not None
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
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM pending_operations")
        count = cursor.rowcount
        self._conn.commit()
        return count
    
    def delete_by_path(self, path: str) -> bool:
        """Delete a record by its canonical path.
        
        Args:
            path: The canonical path of the file.
            
        Returns:
            True if a record was deleted.
        """
        assert self._conn is not None
        cursor = self._conn.cursor()
        
        # First get the hash to clean up source_path_index
        cursor.execute(
            "SELECT similarity_hash FROM media WHERE canonical_path = ?",
            (path,)
        )
        row = cursor.fetchone()
        
        # Delete from media table
        cursor.execute("DELETE FROM media WHERE canonical_path = ?", (path,))
        deleted = cursor.rowcount > 0
        
        # Clean up source_path_index entries pointing to this path
        cursor.execute(
            "DELETE FROM source_path_index WHERE path = ?",
            (path,)
        )
        
        self._conn.commit()
        return deleted
    
    def update_path(self, old_path: str, new_path: str) -> bool:
        """Update a record's canonical path (for rename/move).
        
        Args:
            old_path: Current path.
            new_path: New path.
            
        Returns:
            True if a record was updated.
        """
        assert self._conn is not None
        cursor = self._conn.cursor()
        
        cursor.execute(
            """UPDATE media 
               SET canonical_path = ?, updated_at = CURRENT_TIMESTAMP 
               WHERE canonical_path = ?""",
            (new_path, old_path)
        )
        updated = cursor.rowcount > 0
        
        # Also update source_path_index if the path was there
        cursor.execute(
            "UPDATE source_path_index SET path = ? WHERE path = ?",
            (new_path, old_path)
        )
        
        self._conn.commit()
        return updated
    
    def get_all_by_hash(self, similarity_hash: str) -> list[MediaRecord]:
        """Get all records with a given hash (for finding duplicates).
        
        Args:
            similarity_hash: The hash to search for.
            
        Returns:
            List of all records with this hash.
        """
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM media WHERE similarity_hash = ?",
            (similarity_hash,)
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def get_by_path(self, path: str) -> Optional[MediaRecord]:
        """Get record by canonical path.
        
        Args:
            path: The canonical path.
            
        Returns:
            MediaRecord if found, None otherwise.
        """
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM media WHERE canonical_path = ?",
            (path,)
        )
        row = cursor.fetchone()
        return self._row_to_record(row) if row else None
    
    def get_duplicate_hashes(self) -> list[tuple[str, int]]:
        """Get all hashes that have more than one file.
        
        Returns:
            List of (hash, count) tuples for duplicates.
        """
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT similarity_hash, COUNT(*) as cnt 
            FROM media 
            WHERE similarity_hash IS NOT NULL
            GROUP BY similarity_hash 
            HAVING cnt > 1
            ORDER BY cnt DESC
        """)
        return [(row[0], row[1]) for row in cursor.fetchall()]
    
    def count_all(self) -> int:
        """Count total records in database."""
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM media")
        return cursor.fetchone()[0]
    
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
        
        # Handle optional new columns that may not exist in older databases
        row_dict = dict(row)
        
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
            faces_hashes=row_dict.get("faces_hashes") or "",
            ai_desc=row_dict.get("ai_desc") or "",
            objects=row_dict.get("objects") or "",
        )
    
    def __enter__(self) -> "SQLiteMediaRepository":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
