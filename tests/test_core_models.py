"""Tests for core domain models."""
import pytest
from datetime import datetime
from pathlib import Path

from uncloud.core.models import (
    ProcessingAction,
    DuplicateResolution,
    MediaItem,
    HashResult,
    DuplicateGroup,
    CopyPlan,
    ProcessingResult,
    ProcessingStats,
)


class TestMediaItem:
    """Tests for MediaItem dataclass."""
    
    def test_create_basic(self, tmp_path: Path):
        """Test basic MediaItem creation."""
        test_file = tmp_path / "test.jpg"
        test_file.touch()
        
        item = MediaItem(path=test_file, owner="alice")
        
        assert item.path == test_file
        assert item.owner == "alice"
        assert item.tags == ()
        assert item.sidecar_path is None
        assert item.is_media is True
    
    def test_with_tags(self, tmp_path: Path):
        """Test MediaItem with tags."""
        test_file = tmp_path / "test.jpg"
        test_file.touch()
        
        item = MediaItem(
            path=test_file, 
            owner="bob",
            tags=("vacation", "2024"),
        )
        
        assert item.tags == ("vacation", "2024")
    
    def test_extension_property(self, tmp_path: Path):
        """Test extension property."""
        test_file = tmp_path / "photo.JPEG"
        test_file.touch()
        
        item = MediaItem(path=test_file, owner="alice")
        assert item.extension == ".jpeg"
    
    def test_name_property(self, tmp_path: Path):
        """Test name property."""
        test_file = tmp_path / "my_photo.jpg"
        test_file.touch()
        
        item = MediaItem(path=test_file, owner="alice")
        assert item.name == "my_photo.jpg"
    
    def test_frozen(self, tmp_path: Path):
        """Test that MediaItem is immutable."""
        test_file = tmp_path / "test.jpg"
        test_file.touch()
        
        item = MediaItem(path=test_file, owner="alice")
        
        with pytest.raises(AttributeError):
            item.owner = "bob"


class TestHashResult:
    """Tests for HashResult dataclass."""
    
    def test_create_basic(self, tmp_path: Path):
        """Test basic HashResult creation."""
        test_file = tmp_path / "test.jpg"
        test_file.touch()
        
        item = MediaItem(path=test_file, owner="alice")
        result = HashResult(item=item)
        
        assert result.item == item
        assert result.similarity_hash is None
        assert result.date_taken is None
        assert result.error is None
    
    def test_with_hash(self, tmp_path: Path):
        """Test HashResult with hash value."""
        test_file = tmp_path / "test.jpg"
        test_file.touch()
        
        item = MediaItem(path=test_file, owner="alice")
        result = HashResult(
            item=item,
            similarity_hash="phash:abc123",
            date_taken=datetime(2024, 1, 15, 10, 30),
            date_source="exif",
            width=1920,
            height=1080,
        )
        
        assert result.similarity_hash == "phash:abc123"
        assert result.date_taken == datetime(2024, 1, 15, 10, 30)
        assert result.width == 1920
        assert result.height == 1080
    
    def test_resolution_property(self, tmp_path: Path):
        """Test resolution calculation."""
        test_file = tmp_path / "test.jpg"
        test_file.touch()
        
        item = MediaItem(path=test_file, owner="alice")
        result = HashResult(item=item, width=1920, height=1080)
        
        assert result.resolution == 1920 * 1080
    
    def test_resolution_none(self, tmp_path: Path):
        """Test resolution when dimensions missing."""
        test_file = tmp_path / "test.jpg"
        test_file.touch()
        
        item = MediaItem(path=test_file, owner="alice")
        result = HashResult(item=item)
        
        assert result.resolution == 0
    
    def test_is_valid(self, tmp_path: Path):
        """Test is_valid property."""
        test_file = tmp_path / "test.jpg"
        test_file.touch()
        
        item = MediaItem(path=test_file, owner="alice")
        
        # Valid result
        result = HashResult(item=item, similarity_hash="phash:abc123")
        assert result.is_valid is True
        
        # Invalid - no hash
        result = HashResult(item=item)
        assert result.is_valid is False
        
        # Invalid - has error
        result = HashResult(item=item, similarity_hash="phash:abc123", error="failed")
        assert result.is_valid is False


class TestDuplicateGroup:
    """Tests for DuplicateGroup dataclass."""
    
    def test_create(self, tmp_path: Path):
        """Test DuplicateGroup creation."""
        files = []
        items = []
        for i in range(3):
            f = tmp_path / f"test{i}.jpg"
            f.touch()
            files.append(f)
            items.append(MediaItem(path=f, owner="alice"))
        
        results = tuple(
            HashResult(item=item, similarity_hash="phash:abc123", width=1920, height=1080)
            for item in items
        )
        
        group = DuplicateGroup(hash_value="phash:abc123", items=results)
        
        assert group.hash_value == "phash:abc123"
        assert group.count == 3
    
    def test_get_best_higher_resolution(self, tmp_path: Path):
        """Test getting best item by resolution."""
        items = []
        for i in range(3):
            f = tmp_path / f"test{i}.jpg"
            f.touch()
            items.append(MediaItem(path=f, owner="alice"))
        
        results = (
            HashResult(item=items[0], similarity_hash="h", width=640, height=480),
            HashResult(item=items[1], similarity_hash="h", width=1920, height=1080),
            HashResult(item=items[2], similarity_hash="h", width=800, height=600),
        )
        
        group = DuplicateGroup(hash_value="h", items=results)
        best = group.get_best(DuplicateResolution.KEEP_HIGHER_RESOLUTION)
        
        assert best.width == 1920


class TestProcessingStats:
    """Tests for ProcessingStats."""
    
    def test_initial_state(self):
        """Test initial state is zeros."""
        stats = ProcessingStats()
        
        assert stats.total_files == 0
        assert stats.processed == 0
        assert stats.copied == 0
        assert stats.errors == 0
    
    def test_record_copied(self, tmp_path: Path):
        """Test recording a copied result."""
        stats = ProcessingStats()
        f = tmp_path / "test.jpg"
        f.touch()
        item = MediaItem(path=f, owner="alice")
        
        result = ProcessingResult(item=item, action=ProcessingAction.COPIED)
        stats.record(result)
        
        assert stats.processed == 1
        assert stats.copied == 1
    
    def test_record_error(self, tmp_path: Path):
        """Test recording an error result."""
        stats = ProcessingStats()
        f = tmp_path / "test.jpg"
        f.touch()
        item = MediaItem(path=f, owner="alice")
        
        result = ProcessingResult(item=item, action=ProcessingAction.ERROR, error="failed")
        stats.record(result)
        
        assert stats.processed == 1
        assert stats.errors == 1
    
    def test_alias_properties(self):
        """Test backward compatibility alias properties."""
        stats = ProcessingStats(total_files=100, copied=50, skipped_duplicate=25)
        
        assert stats.files_scanned == 100
        assert stats.files_copied == 50
        assert stats.duplicates_skipped == 25
    
    def test_summary(self):
        """Test summary dict."""
        stats = ProcessingStats(total_files=100, copied=50, errors=5)
        summary = stats.summary()
        
        assert summary["total"] == 100
        assert summary["copied"] == 50
        assert summary["errors"] == 5
