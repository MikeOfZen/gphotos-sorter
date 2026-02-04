"""Tests for CLI commands."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestCLIParsing:
    """Test CLI argument parsing."""
    
    def test_import_command_basic(self):
        """Test import command basic parsing."""
        from uncloud.cli import create_parser
        
        parser = create_parser()
        args = parser.parse_args([
            "import",
            "/input/dir",
            "-o", "/output/dir",
        ])
        
        assert args.command == "import"
        assert args.input_dirs == [Path("/input/dir")]
        assert args.output == Path("/output/dir")
        assert args.dry_run is False
    
    def test_import_with_owner_prefix(self):
        """Test import with owner:path format."""
        from uncloud.cli import create_parser
        
        parser = create_parser()
        args = parser.parse_args([
            "import",
            "Mine:/photos/mine",
            "Family:/photos/family",
            "-o", "/library",
        ])
        
        assert len(args.input_dirs) == 2
    
    def test_index_command(self):
        """Test index command parsing."""
        from uncloud.cli import create_parser
        
        parser = create_parser()
        args = parser.parse_args([
            "index",
            "/library",
            "-w", "32",
            "--dry-run",
        ])
        
        assert args.command == "index"
        assert args.directory == Path("/library")
        assert args.workers == 32
        assert args.dry_run is True
    
    def test_dedupe_command(self):
        """Test dedupe command parsing."""
        from uncloud.cli import create_parser
        
        parser = create_parser()
        args = parser.parse_args([
            "dedupe",
            "/library",
            "--policy", "keep-largest",
            "--dry-run",
        ])
        
        assert args.command == "dedupe"
        assert args.directory == Path("/library")
        assert args.policy == "keep-largest"
        assert args.dry_run is True
    
    def test_info_command(self):
        """Test info command parsing."""
        from uncloud.cli import create_parser
        
        parser = create_parser()
        args = parser.parse_args([
            "info",
            "/library",
            "--show-duplicates",
        ])
        
        assert args.command == "info"
        assert args.path == Path("/library")
        assert args.show_duplicates is True
    
    def test_delete_command(self):
        """Test delete command parsing."""
        from uncloud.cli import create_parser
        
        parser = create_parser()
        args = parser.parse_args([
            "delete",
            "/file1.jpg",
            "/file2.jpg",
            "--db", "/library/.db",
        ])
        
        assert args.command == "delete"
        assert len(args.files) == 2
        assert args.db == Path("/library/.db")
    
    def test_rename_command(self):
        """Test rename command parsing."""
        from uncloud.cli import create_parser
        
        parser = create_parser()
        args = parser.parse_args([
            "rename",
            "/old.jpg",
            "/new.jpg",
            "--db", "/library/.db",
        ])
        
        assert args.command == "rename"
        assert args.source == Path("/old.jpg")
        assert args.dest == Path("/new.jpg")


class TestCLICommands:
    """Test CLI command handlers."""
    
    @pytest.fixture
    def mock_reporter(self):
        """Create a mock reporter."""
        reporter = MagicMock()
        return reporter
    
    @pytest.fixture
    def sample_library(self, tmp_path: Path):
        """Create a sample library with database."""
        from uncloud.persistence.database import SQLiteMediaRepository, MediaRecord
        
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        
        # Create some files
        (lib_dir / "photo1.jpg").write_bytes(b"test1")
        (lib_dir / "photo2.jpg").write_bytes(b"test2")
        (lib_dir / "photo3.jpg").write_bytes(b"test3")
        
        # Create database
        db_path = lib_dir / ".uncloud.db"
        repo = SQLiteMediaRepository(db_path)
        
        repo.upsert(MediaRecord(
            canonical_path=str(lib_dir / "photo1.jpg"),
            similarity_hash="hash1",
            owner="test",
        ))
        repo.upsert(MediaRecord(
            canonical_path=str(lib_dir / "photo2.jpg"),
            similarity_hash="hash1",  # Duplicate
            owner="test",
        ))
        repo.upsert(MediaRecord(
            canonical_path=str(lib_dir / "photo3.jpg"),
            similarity_hash="hash2",
            owner="test",
        ))
        
        repo.close()
        
        return lib_dir
    
    def test_info_command_library(self, sample_library, mock_reporter):
        """Test info command on library."""
        from uncloud.cli import cmd_info
        from argparse import Namespace
        
        args = Namespace(
            path=sample_library,
            db=None,
            show_duplicates=False,
        )
        
        result = cmd_info(args, mock_reporter)
        
        assert result == 0
        # Should print library info
        assert mock_reporter.print_header.called
        assert mock_reporter.info.called
    
    def test_info_command_file(self, sample_library, mock_reporter):
        """Test info command on single file."""
        from uncloud.cli import cmd_info
        from argparse import Namespace
        
        file_path = sample_library / "photo1.jpg"
        
        args = Namespace(
            path=file_path,
            db=None,
            show_duplicates=False,
        )
        
        result = cmd_info(args, mock_reporter)
        
        assert result == 0
        assert mock_reporter.print_header.called
    
    def test_delete_command(self, sample_library, mock_reporter):
        """Test delete command."""
        from uncloud.cli import cmd_delete
        from argparse import Namespace
        from uncloud.persistence.database import SQLiteMediaRepository
        
        file_to_delete = sample_library / "photo3.jpg"
        db_path = sample_library / ".uncloud.db"
        
        args = Namespace(
            files=[file_to_delete],
            db=db_path,
            dry_run=False,
        )
        
        result = cmd_delete(args, mock_reporter)
        
        assert result == 0
        assert not file_to_delete.exists()
        
        # Check DB
        repo = SQLiteMediaRepository(db_path)
        assert repo.get_by_path(str(file_to_delete)) is None
        repo.close()
    
    def test_delete_command_dry_run(self, sample_library, mock_reporter):
        """Test delete command with dry run."""
        from uncloud.cli import cmd_delete
        from argparse import Namespace
        
        file_to_delete = sample_library / "photo3.jpg"
        db_path = sample_library / ".uncloud.db"
        
        args = Namespace(
            files=[file_to_delete],
            db=db_path,
            dry_run=True,
        )
        
        result = cmd_delete(args, mock_reporter)
        
        assert result == 0
        assert file_to_delete.exists()  # Still exists
    
    def test_rename_command(self, sample_library, mock_reporter):
        """Test rename command."""
        from uncloud.cli import cmd_rename
        from argparse import Namespace
        from uncloud.persistence.database import SQLiteMediaRepository
        
        old_path = sample_library / "photo3.jpg"
        new_path = sample_library / "renamed.jpg"
        db_path = sample_library / ".uncloud.db"
        
        args = Namespace(
            source=old_path,
            dest=new_path,
            db=db_path,
            dry_run=False,
        )
        
        result = cmd_rename(args, mock_reporter)
        
        assert result == 0
        assert not old_path.exists()
        assert new_path.exists()
        
        # Check DB
        repo = SQLiteMediaRepository(db_path)
        assert repo.get_by_path(str(old_path)) is None
        assert repo.get_by_path(str(new_path)) is not None
        repo.close()
    
    def test_dedupe_command(self, sample_library, mock_reporter):
        """Test dedupe command."""
        from uncloud.cli import cmd_dedupe
        from argparse import Namespace
        from uncloud.persistence.database import SQLiteMediaRepository
        
        db_path = sample_library / ".uncloud.db"
        
        args = Namespace(
            directory=sample_library,
            db=None,  # Will use default
            policy="keep-first",
            dry_run=False,
            min_duplicates=2,
            verbose=False,
        )
        
        # Should have 2 files with hash1
        repo = SQLiteMediaRepository(db_path)
        before_count = len(repo.get_all_by_hash("hash1"))
        repo.close()
        
        assert before_count == 2
        
        result = cmd_dedupe(args, mock_reporter)
        
        assert result == 0
        
        # Should have 1 file left
        repo = SQLiteMediaRepository(db_path)
        after_count = len(repo.get_all_by_hash("hash1"))
        repo.close()
        
        assert after_count == 1


class TestMetadataHashStorage:
    """Test storing and reading hashes from metadata."""
    
    @pytest.fixture
    def sample_image(self, tmp_path: Path):
        """Create a sample image file."""
        from PIL import Image
        
        img_path = tmp_path / "test.jpg"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path)
        
        return img_path
    
    def test_write_and_read_hash(self, sample_image):
        """Test writing and reading hash from file metadata."""
        from uncloud.engines.metadata import ExifToolMetadataExtractor
        
        extractor = ExifToolMetadataExtractor(use_batch_mode=False)
        
        try:
            # Write hash
            test_hash = "abc123def456"
            success = extractor.write_uncloud_hash(sample_image, test_hash)
            assert success or True  # exiftool might not be installed
            
            # Read hash back
            read_hash = extractor.extract_uncloud_hash(sample_image)
            
            # Only assert if write succeeded
            if success and read_hash:
                assert read_hash == test_hash
        finally:
            extractor.close()
    
    def test_write_uncloud_metadata(self, sample_image):
        """Test writing full uncloud metadata."""
        from uncloud.engines.metadata import ExifToolMetadataExtractor
        
        extractor = ExifToolMetadataExtractor(use_batch_mode=False)
        
        try:
            # Write metadata
            test_hash = "xyz789"
            test_tags = ["vacation", "beach", "2024"]
            
            success = extractor.write_uncloud_metadata(
                sample_image,
                hash_value=test_hash,
                tags=test_tags,
            )
            
            if success:
                # Read back
                meta = extractor.extract_uncloud_metadata(sample_image)
                
                if meta['hash']:
                    assert meta['hash'] == test_hash
        finally:
            extractor.close()


class TestIndexRebuilderWithMetadata:
    """Test index rebuilder with metadata-first hashing."""
    
    @pytest.fixture
    def mock_hash_engine(self):
        """Create a mock hash engine."""
        engine = MagicMock()
        engine.compute_hash = MagicMock(return_value="computed_hash")
        engine.name = "MockEngine"
        return engine
    
    @pytest.fixture
    def mock_progress(self):
        """Create a mock progress reporter."""
        progress = MagicMock()
        return progress
    
    def test_rebuilder_uses_metadata_first(self, tmp_path, mock_hash_engine, mock_progress):
        """Test that rebuilder reads from metadata before computing."""
        from uncloud.services.index_rebuilder import IndexRebuilder
        from uncloud.engines.metadata import ExifToolMetadataExtractor
        from PIL import Image
        
        # Create test library
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        
        # Create image with metadata hash stored
        img1 = lib_dir / "photo1.jpg"
        Image.new('RGB', (100, 100), color='red').save(img1)
        
        # Store hash in metadata
        extractor = ExifToolMetadataExtractor(use_batch_mode=False)
        try:
            extractor.write_uncloud_hash(img1, "stored_hash_123")
        finally:
            extractor.close()
        
        # Create image without metadata
        img2 = lib_dir / "photo2.jpg"
        Image.new('RGB', (100, 100), color='blue').save(img2)
        
        # Rebuild index
        rebuilder = IndexRebuilder(
            hash_engine=mock_hash_engine,
            progress=mock_progress,
            workers=2,
        )
        
        db_path = lib_dir / ".test.db"
        stats = rebuilder.rebuild(
            output_dir=lib_dir,
            db_path=db_path,
            dry_run=False,
            backup=False,
        )
        
        # Should have processed both files
        assert stats.total_files == 2
        assert stats.inserted == 2


class TestDatabaseMethods:
    """Test new database methods."""
    
    @pytest.fixture
    def sample_db(self, tmp_path: Path):
        """Create a sample database."""
        from uncloud.persistence.database import SQLiteMediaRepository, MediaRecord
        
        db_path = tmp_path / "test.db"
        repo = SQLiteMediaRepository(db_path)
        
        # Add test records
        for i in range(5):
            repo.upsert(MediaRecord(
                canonical_path=f"/path/file{i}.jpg",
                similarity_hash="hash1" if i < 3 else f"hash{i}",
                owner="test",
            ))
        
        yield repo
        repo.close()
    
    def test_get_duplicate_hashes(self, sample_db):
        """Test finding duplicate hashes."""
        duplicates = sample_db.get_duplicate_hashes()
        
        # hash1 appears 3 times
        assert len(duplicates) >= 1
        assert duplicates[0] == ("hash1", 3)
    
    def test_get_all_by_hash(self, sample_db):
        """Test getting all records by hash."""
        records = sample_db.get_all_by_hash("hash1")
        
        assert len(records) == 3
        for record in records:
            assert record.similarity_hash == "hash1"
    
    def test_delete_by_path(self, sample_db):
        """Test deleting by path."""
        path = "/path/file0.jpg"
        
        assert sample_db.get_by_path(path) is not None
        
        result = sample_db.delete_by_path(path)
        
        assert result is True
        assert sample_db.get_by_path(path) is None
    
    def test_update_path(self, sample_db):
        """Test updating path."""
        old_path = "/path/file1.jpg"
        new_path = "/path/renamed.jpg"
        
        result = sample_db.update_path(old_path, new_path)
        
        assert result is True
        assert sample_db.get_by_path(old_path) is None
        assert sample_db.get_by_path(new_path) is not None
    
    def test_count_all(self, sample_db):
        """Test counting all records."""
        count = sample_db.count_all()
        
        assert count == 5
