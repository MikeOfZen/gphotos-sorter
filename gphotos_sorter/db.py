from __future__ import annotations

import json
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
                notes TEXT
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_media_hash ON media(similarity_hash)"
        )
        self.connection.commit()

    def get_by_hash(self, similarity_hash: str) -> Optional[MediaRecord]:
        cursor = self.connection.execute(
            "SELECT similarity_hash, canonical_path, owner, date_taken, date_source, tags, source_paths, status, notes FROM media WHERE similarity_hash = ?",
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
        )

    def upsert(self, record: MediaRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO media (
                similarity_hash, canonical_path, owner, date_taken, date_source, tags, source_paths, status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(similarity_hash) DO UPDATE SET
                canonical_path = excluded.canonical_path,
                owner = excluded.owner,
                date_taken = excluded.date_taken,
                date_source = excluded.date_source,
                tags = excluded.tags,
                source_paths = excluded.source_paths,
                status = excluded.status,
                notes = excluded.notes
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
            ),
        )
        self.connection.commit()

    def update_existing(self, similarity_hash: str, tags: Iterable[str], source_paths: Iterable[str]) -> None:
        existing = self.get_by_hash(similarity_hash)
        if not existing:
            return
        merged_tags = sorted(set(existing.tags).union(tags))
        merged_sources = sorted(set(existing.source_paths).union(source_paths))
        self.connection.execute(
            "UPDATE media SET tags = ?, source_paths = ? WHERE similarity_hash = ?",
            (json.dumps(merged_tags), json.dumps(merged_sources), similarity_hash),
        )
        self.connection.commit()

    def has_source_path(self, source_path: str) -> bool:
        """Check if a source path is already recorded in any media entry."""
        cursor = self.connection.execute(
            "SELECT 1 FROM media WHERE source_paths LIKE ? LIMIT 1",
            (f'%"{source_path}"%',),
        )
        return cursor.fetchone() is not None

    def get_all_source_paths(self) -> set[str]:
        """Get all source paths from database for fast lookup."""
        cursor = self.connection.execute("SELECT source_paths FROM media")
        all_paths: set[str] = set()
        for row in cursor:
            if row[0]:
                paths = json.loads(row[0])
                all_paths.update(paths)
        return all_paths
