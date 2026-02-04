"""Unit tests for database module."""
import pytest
import tempfile
from pathlib import Path

from gphotos_sorter.db import MediaDatabase, MediaRecord


class TestMediaDatabase:
    """Tests for MediaDatabase class."""
    
    def setup_method(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.sqlite"
        self.db = MediaDatabase(self.db_path)
        
    def teardown_method(self):
        """Clean up test database."""
        self.db.close()
        if self.db_path.exists():
            self.db_path.unlink()
            
    def test_database_creation(self):
        """Test that database file is created."""
        assert self.db_path.exists()
        
    def test_upsert_new_record(self):
        """Test inserting a new record."""
        record = MediaRecord(
            similarity_hash="abc123",
            canonical_path="/output/photo.jpg",
            owner="Test",
            date_taken="2021-06-15T10:30:00",
            date_source="exif",
            tags=["Album"],
            source_paths=["/input/photo.jpg"],
            status="ok",
            notes=None,
        )
        self.db.upsert(record)
        
        result = self.db.get_by_hash("abc123")
        assert result is not None
        assert result.canonical_path == "/output/photo.jpg"
        
    def test_upsert_update_existing(self):
        """Test updating an existing record."""
        record1 = MediaRecord(
            similarity_hash="abc123",
            canonical_path="/output/photo.jpg",
            owner="Test",
            date_taken="2021-06-15T10:30:00",
            date_source="exif",
            tags=["Album1"],
            source_paths=["/input/photo1.jpg"],
            status="ok",
            notes=None,
        )
        self.db.upsert(record1)
        
        record2 = MediaRecord(
            similarity_hash="abc123",
            canonical_path="/output/photo.jpg",
            owner="Test",
            date_taken="2021-06-15T10:30:00",
            date_source="exif",
            tags=["Album2"],
            source_paths=["/input/photo2.jpg"],
            status="ok",
            notes=None,
        )
        self.db.upsert(record2)
        
        result = self.db.get_by_hash("abc123")
        assert result is not None
        # Should have updated to new values
        
    def test_get_by_hash_not_found(self):
        """Test getting a non-existent record."""
        result = self.db.get_by_hash("nonexistent")
        assert result is None
        
    def test_update_existing(self):
        """Test update_existing method."""
        record = MediaRecord(
            similarity_hash="abc123",
            canonical_path="/output/photo.jpg",
            owner="Test",
            date_taken="2021-06-15T10:30:00",
            date_source="exif",
            tags=["Album1"],
            source_paths=["/input/photo1.jpg"],
            status="ok",
            notes=None,
        )
        self.db.upsert(record)
        
        self.db.update_existing("abc123", ["Album2"], ["/input/photo2.jpg"])
        
        result = self.db.get_by_hash("abc123")
        assert "Album1" in result.tags
        assert "Album2" in result.tags
        assert "/input/photo1.jpg" in result.source_paths
        assert "/input/photo2.jpg" in result.source_paths
        
    def test_get_all_source_paths(self):
        """Test getting all source paths."""
        record1 = MediaRecord(
            similarity_hash="abc123",
            canonical_path="/output/photo1.jpg",
            owner="Test",
            date_taken=None,
            date_source="missing",
            tags=[],
            source_paths=["/input/photo1.jpg", "/input/photo1_dup.jpg"],
            status="ok",
            notes=None,
        )
        record2 = MediaRecord(
            similarity_hash="def456",
            canonical_path="/output/photo2.jpg",
            owner="Test",
            date_taken=None,
            date_source="missing",
            tags=[],
            source_paths=["/input/photo2.jpg"],
            status="ok",
            notes=None,
        )
        self.db.upsert(record1)
        self.db.upsert(record2)
        
        all_sources = self.db.get_all_source_paths()
        assert "/input/photo1.jpg" in all_sources
        assert "/input/photo1_dup.jpg" in all_sources
        assert "/input/photo2.jpg" in all_sources

    def test_has_source_path_index(self):
        """Test source_path_index lookups and updates."""
        record = MediaRecord(
            similarity_hash="abc123",
            canonical_path="/output/photo.jpg",
            owner="Test",
            date_taken=None,
            date_source="missing",
            tags=[],
            source_paths=["/input/photo.jpg"],
            status="ok",
            notes=None,
        )
        self.db.upsert(record)
        assert self.db.has_source_path("/input/photo.jpg")
        assert not self.db.has_source_path("/input/other.jpg")

        self.db.update_existing("abc123", [], ["/input/other.jpg"])
        assert self.db.has_source_path("/input/other.jpg")

    def test_pending_operations(self):
        """Test pending operations lifecycle."""
        op_id = self.db.add_pending_operation(
            "/input/photo.jpg",
            "/output/photo.jpg",
            "abc123",
            "copy",
        )
        pending = self.db.get_pending_operations()
        assert len(pending) == 1
        assert pending[0].source_path == "/input/photo.jpg"

        self.db.complete_pending_operation(op_id)
        pending = self.db.get_pending_operations()
        assert pending == []

        op_id2 = self.db.add_pending_operation(
            "/input/photo2.jpg",
            "/output/photo2.jpg",
            "def456",
            "copy",
        )
        assert op_id2
        cleared = self.db.clear_all_pending_operations()
        assert cleared == 1
        
    def test_record_retrieval(self):
        """Test retrieving a record by hash."""
        record = MediaRecord(
            similarity_hash="abc123",
            canonical_path="/output/photo.jpg",
            owner="Test",
            date_taken=None,
            date_source="missing",
            tags=[],
            source_paths=["/input/photo.jpg"],
            status="ok",
            notes=None,
        )
        self.db.upsert(record)
        
        result = self.db.get_by_hash("abc123")
        assert result is not None
        assert result.similarity_hash == "abc123"


class TestMediaRecord:
    """Tests for MediaRecord dataclass."""
    
    def test_record_creation(self):
        """Test creating a media record."""
        record = MediaRecord(
            similarity_hash="abc123",
            canonical_path="/output/photo.jpg",
            owner="Test",
            date_taken="2021-06-15T10:30:00",
            date_source="exif",
            tags=["Album"],
            source_paths=["/input/photo.jpg"],
            status="ok",
            notes="test note",
        )
        assert record.similarity_hash == "abc123"
        assert record.owner == "Test"
        assert record.status == "ok"
        
    def test_record_with_none_values(self):
        """Test record with None values."""
        record = MediaRecord(
            similarity_hash="abc123",
            canonical_path="/output/photo.jpg",
            owner="Test",
            date_taken=None,
            date_source="missing",
            tags=[],
            source_paths=[],
            status="missing_date",
            notes=None,
        )
        assert record.date_taken is None
        assert record.notes is None
