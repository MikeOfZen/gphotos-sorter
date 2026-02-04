"""Tests for SQLite media repository."""
import pytest
from pathlib import Path
from datetime import datetime

from uncloud.persistence.database import (
    SQLiteMediaRepository,
    MediaRecord,
    PendingOperation,
)


class TestSQLiteMediaRepository:
    """Tests for SQLite repository."""
    
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        """Get a test database path."""
        return tmp_path / "test.db"
    
    @pytest.fixture
    def repo(self, db_path: Path):
        """Create a repository instance."""
        repo = SQLiteMediaRepository(db_path)
        yield repo
        repo.close()
    
    def test_create_database(self, db_path):
        """Test database is created."""
        repo = SQLiteMediaRepository(db_path)
        assert db_path.exists()
        repo.close()
    
    def test_upsert_and_get(self, repo):
        """Test inserting and retrieving a record."""
        record = MediaRecord(
            canonical_path="/output/2024/01/photo.jpg",
            similarity_hash="phash:abc123",
            owner="alice",
            date_taken=datetime(2024, 1, 15),
            tags="vacation,family",
            width=1920,
            height=1080,
            source_paths="/input/photo.jpg",
        )
        
        repo.upsert(record)
        
        # Retrieve by hash
        retrieved = repo.get_by_hash("phash:abc123")
        
        assert retrieved is not None
        assert retrieved.canonical_path == "/output/2024/01/photo.jpg"
        assert retrieved.owner == "alice"
        assert retrieved.width == 1920
    
    def test_get_nonexistent(self, repo):
        """Test getting nonexistent record returns None."""
        result = repo.get_by_hash("nonexistent")
        assert result is None
    
    def test_has_source_path(self, repo):
        """Test source path tracking."""
        record = MediaRecord(
            canonical_path="/output/photo.jpg",
            similarity_hash="phash:xyz",
            source_paths="/input/photo1.jpg,/input/photo2.jpg",
        )
        repo.upsert(record)
        
        assert repo.has_source_path("/input/photo1.jpg") is True
        assert repo.has_source_path("/input/photo2.jpg") is True
        assert repo.has_source_path("/input/other.jpg") is False
    
    def test_get_all_source_paths(self, repo):
        """Test getting all source paths."""
        record = MediaRecord(
            canonical_path="/output/photo.jpg",
            similarity_hash="phash:123",
            source_paths="/input/a.jpg,/input/b.jpg",
        )
        repo.upsert(record)
        
        paths = repo.get_all_source_paths()
        
        assert "/input/a.jpg" in paths
        assert "/input/b.jpg" in paths
    
    def test_pending_operations(self, repo):
        """Test pending operation tracking."""
        # Add pending operation
        op_id = repo.add_pending_operation(
            source="/input/photo.jpg",
            target="/output/photo.jpg",
            hash_val="phash:abc",
            op="copy",
        )
        
        assert op_id > 0
        
        # Get pending operations
        pending = repo.get_pending_operations()
        assert len(pending) == 1
        assert pending[0].source_path == "/input/photo.jpg"
        assert pending[0].operation == "copy"
        
        # Complete operation
        repo.complete_pending_operation(op_id)
        
        # Should be empty now
        pending = repo.get_pending_operations()
        assert len(pending) == 0
    
    def test_clear_all_pending(self, repo):
        """Test clearing all pending operations."""
        repo.add_pending_operation("/a", "/b", "h1", "copy")
        repo.add_pending_operation("/c", "/d", "h2", "copy")
        
        count = repo.clear_all_pending_operations()
        
        assert count == 2
        assert len(repo.get_pending_operations()) == 0
    
    def test_upsert_update(self, repo):
        """Test upserting updates existing record."""
        record1 = MediaRecord(
            canonical_path="/output/photo.jpg",
            similarity_hash="phash:orig",
            width=640,
        )
        repo.upsert(record1)
        
        # Update with new data
        record2 = MediaRecord(
            canonical_path="/output/photo.jpg",
            similarity_hash="phash:new",
            width=1920,
        )
        repo.upsert(record2)
        
        # Should have updated
        retrieved = repo.get_by_hash("phash:new")
        assert retrieved is not None
        assert retrieved.width == 1920
    
    def test_context_manager(self, db_path):
        """Test repository as context manager."""
        with SQLiteMediaRepository(db_path) as repo:
            record = MediaRecord(
                canonical_path="/test.jpg",
                similarity_hash="phash:test",
            )
            repo.upsert(record)
            result = repo.get_by_hash("phash:test")
            assert result is not None


class TestMediaRecord:
    """Tests for MediaRecord dataclass."""
    
    def test_create_minimal(self):
        """Test minimal record creation."""
        record = MediaRecord(canonical_path="/path/to/file.jpg")
        
        assert record.canonical_path == "/path/to/file.jpg"
        assert record.similarity_hash is None
        assert record.owner == ""
    
    def test_create_full(self):
        """Test full record creation."""
        record = MediaRecord(
            canonical_path="/path/to/file.jpg",
            similarity_hash="phash:abc",
            owner="alice",
            date_taken=datetime(2024, 1, 15),
            tags="tag1,tag2",
            width=1920,
            height=1080,
            source_paths="/input/file.jpg",
        )
        
        assert record.similarity_hash == "phash:abc"
        assert record.owner == "alice"
        assert record.width == 1920


class TestPendingOperation:
    """Tests for PendingOperation dataclass."""
    
    def test_create(self):
        """Test creating a pending operation."""
        op = PendingOperation(
            id=1,
            source_path="/input/file.jpg",
            target_path="/output/file.jpg",
            similarity_hash="phash:abc",
            operation="copy",
            created_at=datetime.now(),
        )
        
        assert op.id == 1
        assert op.source_path == "/input/file.jpg"
        assert op.operation == "copy"
