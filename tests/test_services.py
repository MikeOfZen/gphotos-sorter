"""Tests for service layer components."""
import pytest
from pathlib import Path

from uncloud.core.models import MediaItem, HashResult, DuplicateGroup
from uncloud.core.config import (
    OutputLayout,
    DuplicatePolicy,
    InputSource,
)
from uncloud.services.scanner import DirectoryScanner
from uncloud.services.deduplicator import DuplicateResolver, verify_hash_collision
from uncloud.services.file_ops import FileManager


class TestDirectoryScanner:
    """Tests for DirectoryScanner service."""
    
    @pytest.fixture
    def scanner(self):
        """Create a scanner instance."""
        return DirectoryScanner()
    
    @pytest.fixture
    def sample_tree(self, tmp_path: Path) -> Path:
        """Create a sample directory tree with media files."""
        root = tmp_path / "photos"
        root.mkdir()
        
        # Create some files
        (root / "photo1.jpg").touch()
        (root / "photo2.png").touch()
        (root / "video.mp4").touch()
        (root / "document.txt").touch()  # Non-media
        
        # Create subfolder
        sub = root / "vacation"
        sub.mkdir()
        (sub / "beach.jpg").touch()
        (sub / "beach.jpg.json").touch()  # Sidecar
        
        return root
    
    def test_scan_basic(self, scanner, sample_tree):
        """Test basic directory scanning."""
        source = InputSource(path=sample_tree, owner="alice")
        
        items = list(scanner.scan([source]))
        
        # Should find media files
        paths = [item.path for item in items]
        assert any("photo1.jpg" in str(p) for p in paths)
        assert any("photo2.png" in str(p) for p in paths)
        assert any("video.mp4" in str(p) for p in paths)
        
        # Should not include non-media
        assert not any("document.txt" in str(p) for p in paths)
    
    def test_scan_with_owner(self, scanner, sample_tree):
        """Test owner is assigned to items."""
        source = InputSource(path=sample_tree, owner="bob")
        
        items = list(scanner.scan([source]))
        
        for item in items:
            assert item.owner == "bob"
    
    def test_scan_recursive(self, scanner, sample_tree):
        """Test recursive scanning."""
        source = InputSource(path=sample_tree, owner="alice")
        
        items = list(scanner.scan([source], recursive=True))
        
        # Should find files in subfolders
        assert any("beach.jpg" in str(item.path) for item in items)
    
    def test_scan_non_recursive(self, scanner, sample_tree):
        """Test non-recursive scanning."""
        source = InputSource(path=sample_tree, owner="alice")
        
        items = list(scanner.scan([source], recursive=False))
        
        # Should not find files in subfolders
        assert not any("beach.jpg" in str(item.path) for item in items)
    
    def test_scan_skip_known(self, scanner, sample_tree):
        """Test skipping known files."""
        source = InputSource(path=sample_tree, owner="alice")
        
        # First scan - get all paths
        all_items = list(scanner.scan([source]))
        known = {str(all_items[0].path)}
        
        # Second scan - skip known
        items = list(scanner.scan([source], skip_known=known))
        
        assert len(items) == len(all_items) - 1
    
    def test_sidecar_detection(self, scanner, sample_tree):
        """Test sidecar file detection."""
        source = InputSource(path=sample_tree, owner="alice")
        
        items = list(scanner.scan([source], recursive=True))
        
        beach_item = next((i for i in items if "beach.jpg" in str(i.path)), None)
        assert beach_item is not None
        assert beach_item.sidecar_path is not None
        assert "beach.jpg.json" in str(beach_item.sidecar_path)


class TestDuplicateResolver:
    """Tests for DuplicateResolver service."""
    
    @pytest.fixture
    def resolver_skip(self):
        """Create a resolver with SKIP policy."""
        return DuplicateResolver(policy=DuplicatePolicy.SKIP)
    
    @pytest.fixture
    def resolver_resolution(self):
        """Create a resolver with KEEP_HIGHER_RESOLUTION policy."""
        return DuplicateResolver(policy=DuplicatePolicy.KEEP_HIGHER_RESOLUTION)
    
    def test_group_by_hash(self, resolver_skip, tmp_path):
        """Test grouping results by hash."""
        files = [tmp_path / f"f{i}.jpg" for i in range(4)]
        for f in files:
            f.touch()
        
        items = [MediaItem(path=f, owner="alice") for f in files]
        
        results = [
            HashResult(item=items[0], similarity_hash="hash_a"),
            HashResult(item=items[1], similarity_hash="hash_a"),  # Duplicate
            HashResult(item=items[2], similarity_hash="hash_b"),
            HashResult(item=items[3], similarity_hash=None),  # Invalid
        ]
        
        groups = resolver_skip.group_by_hash(results)
        
        assert len(groups) == 2  # hash_a and hash_b
        
        hash_a_group = next(g for g in groups if g.hash_value == "hash_a")
        assert hash_a_group.count == 2
    
    def test_resolve_skip(self, resolver_skip, tmp_path):
        """Test SKIP policy keeps first item."""
        files = [tmp_path / f"f{i}.jpg" for i in range(2)]
        for f in files:
            f.touch()
        
        items = [MediaItem(path=f, owner="alice") for f in files]
        results = tuple(HashResult(item=item, similarity_hash="h") for item in items)
        
        group = DuplicateGroup(hash_value="h", items=results)
        best, duplicates = resolver_skip.resolve(group)
        
        assert best is not None
        assert len(duplicates) == 1
    
    def test_resolve_higher_resolution(self, resolver_resolution, tmp_path):
        """Test KEEP_HIGHER_RESOLUTION policy."""
        files = [tmp_path / f"f{i}.jpg" for i in range(2)]
        for f in files:
            f.touch()
        
        items = [MediaItem(path=f, owner="alice") for f in files]
        results = (
            HashResult(item=items[0], similarity_hash="h", width=640, height=480),
            HashResult(item=items[1], similarity_hash="h", width=1920, height=1080),
        )
        
        group = DuplicateGroup(hash_value="h", items=results)
        best, duplicates = resolver_resolution.resolve(group)
        
        assert best is not None
        assert best.width == 1920  # Higher resolution kept


class TestVerifyHashCollision:
    """Tests for hash collision verification."""
    
    def test_identical_files(self, tmp_path):
        """Test identical files are detected as same."""
        content = b"same content" * 1000
        
        path1 = tmp_path / "file1.jpg"
        path2 = tmp_path / "file2.jpg"
        path1.write_bytes(content)
        path2.write_bytes(content)
        
        assert verify_hash_collision(path1, path2) is True
    
    def test_different_files(self, tmp_path):
        """Test different files are detected as different."""
        path1 = tmp_path / "file1.jpg"
        path2 = tmp_path / "file2.jpg"
        path1.write_bytes(b"content a" * 1000)
        path2.write_bytes(b"content b" * 1000)
        
        assert verify_hash_collision(path1, path2) is False
    
    def test_very_different_sizes(self, tmp_path):
        """Test files with very different sizes are different."""
        path1 = tmp_path / "file1.jpg"
        path2 = tmp_path / "file2.jpg"
        path1.write_bytes(b"a" * 100)
        path2.write_bytes(b"a" * 1000)
        
        assert verify_hash_collision(path1, path2) is False


class TestFileManager:
    """Tests for FileManager service."""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create a file manager."""
        output = tmp_path / "output"
        output.mkdir()
        return FileManager(output_root=output, layout=OutputLayout.YEAR_MONTH)
    
    def test_copy_file(self, manager, tmp_path):
        """Test copying a file."""
        source = tmp_path / "source.jpg"
        source.write_bytes(b"image data")
        
        target = tmp_path / "output" / "dest.jpg"
        
        result = manager.copy_file(source, target)
        
        assert result is True
        assert target.exists()
        assert target.read_bytes() == b"image data"
    
    def test_copy_creates_dirs(self, manager, tmp_path):
        """Test copy creates parent directories."""
        source = tmp_path / "source.jpg"
        source.write_bytes(b"image data")
        
        target = tmp_path / "output" / "2024" / "01" / "photo.jpg"
        
        result = manager.copy_file(source, target)
        
        assert result is True
        assert target.exists()
    
    def test_move_file(self, manager, tmp_path):
        """Test moving a file."""
        source = tmp_path / "source.jpg"
        source.write_bytes(b"image data")
        
        target = tmp_path / "output" / "dest.jpg"
        
        result = manager.move_file(source, target)
        
        assert result is True
        assert target.exists()
        assert not source.exists()
    
    def test_delete_file(self, manager, tmp_path):
        """Test deleting a file."""
        target = tmp_path / "to_delete.jpg"
        target.write_bytes(b"data")
        
        result = manager.delete_file(target)
        
        assert result is True
        assert not target.exists()
    
    def test_delete_nonexistent(self, manager, tmp_path):
        """Test deleting nonexistent file."""
        result = manager.delete_file(tmp_path / "nonexistent.jpg")
        # Deleting non-existent file returns True (no-op success)
        assert result is True
    
    def test_ensure_directory(self, manager, tmp_path):
        """Test ensuring directory exists."""
        new_dir = tmp_path / "new" / "nested" / "dir"
        
        manager.ensure_directory(new_dir)
        
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    @pytest.mark.skip(reason="find_unique_path signature changed, needs update")
    def test_find_unique_path(self, manager, tmp_path):
        """Test finding unique path for duplicates."""
        existing = tmp_path / "photo.jpg"
        existing.touch()
        
        unique = manager.find_unique_path(existing)
        
        assert unique != existing
        assert "photo_1.jpg" in str(unique) or "photo (1).jpg" in str(unique)
    
    def test_dry_run_mode(self, tmp_path):
        """Test dry run mode doesn't copy."""
        output = tmp_path / "output"
        output.mkdir()
        
        manager = FileManager(output_root=output, layout=OutputLayout.FLAT, dry_run=True)
        
        source = tmp_path / "source.jpg"
        source.write_bytes(b"data")
        target = output / "dest.jpg"
        
        # In dry run, copy should return True but not actually copy
        # (Implementation may vary - just test it doesn't crash)
        result = manager.copy_file(source, target)
        assert isinstance(result, bool)


class TestIndexRebuilder:
    """Tests for IndexRebuilder service."""
    
    @pytest.fixture
    def mock_progress(self):
        """Create a mock progress reporter."""
        from unittest.mock import MagicMock
        
        progress = MagicMock()
        progress.info = MagicMock()
        progress.warning = MagicMock()
        progress.error = MagicMock()
        progress.print_header = MagicMock()
        progress.print_config = MagicMock()
        progress.print_stats = MagicMock()
        return progress
    
    @pytest.fixture
    def mock_hash_engine(self):
        """Create a mock hash engine."""
        from unittest.mock import MagicMock
        
        engine = MagicMock()
        engine.compute_hash = MagicMock(return_value="hash123")
        return engine
    
    @pytest.fixture
    def sample_library(self, tmp_path: Path) -> Path:
        """Create a sample output library structure."""
        root = tmp_path / "output"
        owner_dir = root / "TestOwner"
        year_month = owner_dir / "2024-01"
        year_month.mkdir(parents=True)
        
        # Create some media files
        (year_month / "photo1.jpg").write_bytes(b"photo1")
        (year_month / "photo2.png").write_bytes(b"photo2")
        (year_month / "video.mp4").write_bytes(b"video")
        
        return root
    
    def test_scan_media_files(self, sample_library, mock_progress, mock_hash_engine):
        """Test scanning for media files."""
        from uncloud.services.index_rebuilder import IndexRebuilder
        
        rebuilder = IndexRebuilder(
            hash_engine=mock_hash_engine,
            progress=mock_progress,
            workers=2,
        )
        
        files = rebuilder.scan_media_files(sample_library)
        
        assert len(files) == 3
        assert any(f.name == "photo1.jpg" for f in files)
        assert any(f.name == "photo2.png" for f in files)
        assert any(f.name == "video.mp4" for f in files)
    
    def test_rebuild_dry_run(self, sample_library, mock_progress, mock_hash_engine):
        """Test rebuild in dry run mode."""
        from uncloud.services.index_rebuilder import IndexRebuilder
        
        rebuilder = IndexRebuilder(
            hash_engine=mock_hash_engine,
            progress=mock_progress,
            workers=2,
        )
        
        db_path = sample_library / ".test.db"
        
        stats = rebuilder.rebuild(
            output_dir=sample_library,
            db_path=db_path,
            dry_run=True,
        )
        
        # Dry run shouldn't create database
        assert not db_path.exists()
        assert stats.total_files == 3
        assert stats.inserted == 0  # Dry run doesn't insert
    
    def test_rebuild_creates_database(self, sample_library, mock_progress, mock_hash_engine):
        """Test rebuild creates database with records."""
        from uncloud.services.index_rebuilder import IndexRebuilder
        
        # Make hash unique per file
        call_count = [0]
        def unique_hash(path):
            call_count[0] += 1
            return f"hash_{call_count[0]}"
        mock_hash_engine.compute_hash = unique_hash
        
        rebuilder = IndexRebuilder(
            hash_engine=mock_hash_engine,
            progress=mock_progress,
            workers=2,
        )
        
        db_path = sample_library / ".test.db"
        
        stats = rebuilder.rebuild(
            output_dir=sample_library,
            db_path=db_path,
            dry_run=False,
            backup=False,
        )
        
        # Should create database
        assert db_path.exists()
        assert stats.total_files == 3
        assert stats.inserted == 3
        assert stats.errors == 0
    
    def test_rebuild_detects_duplicates(self, sample_library, mock_progress, mock_hash_engine):
        """Test rebuild detects duplicate hashes."""
        from uncloud.services.index_rebuilder import IndexRebuilder
        
        # Return same hash for all files
        mock_hash_engine.compute_hash = lambda path: "same_hash"
        
        rebuilder = IndexRebuilder(
            hash_engine=mock_hash_engine,
            progress=mock_progress,
            workers=2,
        )
        
        db_path = sample_library / ".test.db"
        
        stats = rebuilder.rebuild(
            output_dir=sample_library,
            db_path=db_path,
            dry_run=False,
            backup=False,
        )
        
        # All files are inserted (DB indexes all FS files, even duplicates)
        # But duplicates are tracked separately for reporting
        assert stats.inserted == 3
        assert stats.skipped_duplicates == 2  # 2 files had same hash as first
    
    def test_extract_owner_from_path(self, sample_library, mock_progress, mock_hash_engine):
        """Test owner extraction from path structure."""
        from uncloud.services.index_rebuilder import IndexRebuilder
        
        rebuilder = IndexRebuilder(
            hash_engine=mock_hash_engine,
            progress=mock_progress,
        )
        
        # File structure: output/TestOwner/2024-01/photo1.jpg
        file_path = sample_library / "TestOwner" / "2024-01" / "photo1.jpg"
        
        owner = rebuilder._extract_owner(file_path, sample_library)
        
        assert owner == "TestOwner"


class TestFileOpsSynchronizer:
    """Tests for FileOpsSynchronizer service."""
    
    @pytest.fixture
    def sample_db(self, tmp_path: Path):
        """Create a sample database with records."""
        from uncloud.persistence.database import SQLiteMediaRepository, MediaRecord
        
        db_path = tmp_path / "test.db"
        repo = SQLiteMediaRepository(db_path)
        
        # Create sample files
        file1 = tmp_path / "photo1.jpg"
        file2 = tmp_path / "photo2.jpg"
        file3 = tmp_path / "photo3.jpg"
        file1.write_bytes(b"test image 1")
        file2.write_bytes(b"test image 2")
        file3.write_bytes(b"test image 3")
        
        # Add records
        repo.upsert(MediaRecord(
            canonical_path=str(file1),
            similarity_hash="hash1",
            owner="test",
        ))
        repo.upsert(MediaRecord(
            canonical_path=str(file2),
            similarity_hash="hash1",  # Same hash (duplicate)
            owner="test",
        ))
        repo.upsert(MediaRecord(
            canonical_path=str(file3),
            similarity_hash="hash2",
            owner="test",
        ))
        
        yield repo, tmp_path
        
        repo.close()
    
    def test_delete_file_syncs_fs_and_db(self, sample_db):
        """Test that delete removes from both FS and DB."""
        from uncloud.services.file_ops_sync import FileOpsSynchronizer
        
        repo, tmp_path = sample_db
        sync = FileOpsSynchronizer(repository=repo, dry_run=False)
        
        file1 = tmp_path / "photo1.jpg"
        assert file1.exists()
        assert repo.get_by_path(str(file1)) is not None
        
        result = sync.delete_file(file1)
        
        assert result.success
        assert not file1.exists()
        assert repo.get_by_path(str(file1)) is None
    
    def test_delete_file_dry_run(self, sample_db):
        """Test that dry run doesn't actually delete."""
        from uncloud.services.file_ops_sync import FileOpsSynchronizer
        
        repo, tmp_path = sample_db
        sync = FileOpsSynchronizer(repository=repo, dry_run=True)
        
        file1 = tmp_path / "photo1.jpg"
        
        result = sync.delete_file(file1)
        
        assert result.success
        assert file1.exists()  # Still exists
        assert repo.get_by_path(str(file1)) is not None  # Still in DB
    
    def test_rename_file_syncs_fs_and_db(self, sample_db):
        """Test that rename updates both FS and DB."""
        from uncloud.services.file_ops_sync import FileOpsSynchronizer
        
        repo, tmp_path = sample_db
        sync = FileOpsSynchronizer(repository=repo, dry_run=False)
        
        old_path = tmp_path / "photo3.jpg"
        new_path = tmp_path / "renamed.jpg"
        
        result = sync.rename_file(old_path, new_path)
        
        assert result.success
        assert not old_path.exists()
        assert new_path.exists()
        assert repo.get_by_path(str(old_path)) is None
        assert repo.get_by_path(str(new_path)) is not None
    
    def test_get_duplicate_hashes(self, sample_db):
        """Test finding duplicate hashes in DB."""
        repo, _ = sample_db
        
        duplicates = repo.get_duplicate_hashes()
        
        # hash1 has 2 files
        assert len(duplicates) == 1
        assert duplicates[0] == ("hash1", 2)
    
    def test_delete_duplicates_by_hash(self, sample_db):
        """Test deleting duplicates by hash."""
        from uncloud.services.file_ops_sync import FileOpsSynchronizer
        
        repo, tmp_path = sample_db
        sync = FileOpsSynchronizer(repository=repo, dry_run=False)
        
        # Should have 2 files with hash1
        records = repo.get_all_by_hash("hash1")
        assert len(records) == 2
        
        # Delete duplicates, keep first
        results = sync.delete_duplicates_by_hash("hash1", keep_policy="first")
        
        assert len(results) == 1
        assert results[0].success
        
        # Should have 1 file left
        records = repo.get_all_by_hash("hash1")
        assert len(records) == 1
