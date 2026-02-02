"""Integration tests for the full processing pipeline."""
import pytest
import logging
from datetime import datetime
from pathlib import Path

from gphotos_sorter.config import AppConfig, InputRoot, StorageLayout, FilenameFormat
from gphotos_sorter.scanner import process_media
from .fixtures import (
    FixtureManager,
    ImageWithSidecar,
    ImageNoDate,
    ImageWithDateFolder,
    NonMediaFile,
    create_google_photos_structure,
)


class TestProcessMedia:
    """Integration tests for process_media function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fixture_manager = FixtureManager()
        self.input_dir, self.output_dir = self.fixture_manager.setup()
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.DEBUG)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        self.fixture_manager.teardown()
        
    def test_basic_processing(self):
        """Test basic file processing."""
        # Create a simple fixture
        fixture = ImageWithSidecar(
            name="test_photo",
            date_taken=datetime(2021, 6, 15, 10, 30, 0),
        )
        self.fixture_manager.add_fixture(fixture)
        
        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
            storage_layout=StorageLayout.year_dash_month,
        )
        
        stats = process_media(config, self.logger)
        
        assert stats["processed"] == 1
        assert stats["errors"] == 0
        
        # Check output exists
        expected_folder = self.output_dir / "Test" / fixture.expected_output_folder()
        assert expected_folder.exists()
        
        # Check file was created
        files = list(expected_folder.glob("*.jpg"))
        assert len(files) == 1
        
    def test_album_tags_in_filename(self):
        """Test that album names appear in filename."""
        fixture = ImageWithSidecar(
            name="album_photo",
            parent_folder="VacationAlbum",
            date_taken=datetime(2021, 6, 15, 10, 30, 0),
        )
        self.fixture_manager.add_fixture(fixture)
        
        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
        )
        
        stats = process_media(config, self.logger)
        
        assert stats["processed"] == 1
        
        # Find the output file
        output_folder = self.output_dir / "Test" / fixture.expected_output_folder()
        files = list(output_folder.glob("*.jpg"))
        assert len(files) == 1
        
        # Check album name in filename
        filename = files[0].name
        assert "VacationAlbum" in filename
        
    def test_unknown_folder_for_no_date(self):
        """Test that files without dates go to unknown folder."""
        fixture = ImageNoDate(
            name="no_date_photo",
            parent_folder="RandomAlbum",
        )
        self.fixture_manager.add_fixture(fixture)
        
        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
        )
        
        stats = process_media(config, self.logger)
        
        assert stats["processed"] == 1
        
        # Check file is in unknown folder
        unknown_folder = self.output_dir / "Test" / "unknown"
        assert unknown_folder.exists()
        
        files = list(unknown_folder.glob("*.jpg"))
        assert len(files) == 1
        
    def test_non_media_handling(self):
        """Test that non-media files are copied to non_media folder."""
        fixture = NonMediaFile(
            name="config",
            extension=".txt",
            content="test content",
        )
        self.fixture_manager.add_fixture(fixture)
        
        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
            copy_non_media=True,
        )
        
        stats = process_media(config, self.logger)
        
        assert stats["non_media_copied"] == 1
        
        # Check file is in non_media folder
        non_media_folder = self.output_dir / "Test" / "non_media"
        assert non_media_folder.exists()
        
    def test_skip_non_media(self):
        """Test that non-media files are skipped when configured."""
        fixture = NonMediaFile(
            name="config",
            extension=".txt",
            content="test content",
        )
        self.fixture_manager.add_fixture(fixture)
        
        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
            copy_non_media=False,
        )
        
        stats = process_media(config, self.logger)
        
        assert stats["non_media_copied"] == 0
        
    def test_dry_run_no_files_created(self):
        """Test that dry run doesn't create files."""
        fixture = ImageWithSidecar(
            name="test_photo",
            date_taken=datetime(2021, 6, 15, 10, 30, 0),
        )
        self.fixture_manager.add_fixture(fixture)
        
        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
            dry_run=True,
        )
        
        stats = process_media(config, self.logger)
        
        assert stats["processed"] == 1
        
        # Check no output was created
        owner_folder = self.output_dir / "Test"
        assert not owner_folder.exists()
        
    def test_limit_processing(self):
        """Test file limit."""
        # Create multiple DISTINCT fixtures (different colors for different hashes)
        from PIL import Image
        import tempfile
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, color in enumerate(colors):
            # Create distinct image directly
            img_path = self.input_dir / f"photo_{i}.jpg"
            img = Image.new("RGB", (100, 100), color=color)
            # Add some variety
            for x in range(i * 10, i * 10 + 10):
                for y in range(10):
                    img.putpixel((x, y), (i * 50, 255 - i * 50, i * 30))
            img.save(img_path)
            
            # Create sidecar with distinct time
            sidecar_path = self.input_dir / f"photo_{i}.jpg.json"
            import json
            sidecar = {
                "photoTakenTime": {
                    "timestamp": str(1623732300 + i * 60)  # Different times
                }
            }
            with open(sidecar_path, "w") as f:
                json.dump(sidecar, f)
        
        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
        )
        
        stats = process_media(config, self.logger, limit=3)
        
        assert stats["processed"] == 3
        
    def test_filename_format_options(self):
        """Test custom filename format."""
        fixture = ImageWithSidecar(
            name="formatted_photo",
            parent_folder="Album",
            date_taken=datetime(2021, 6, 15, 10, 30, 45),  # Tuesday
        )
        self.fixture_manager.add_fixture(fixture)
        
        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
            filename_format=FilenameFormat(
                include_time=False,
                max_tags=1,
            ),
        )
        
        stats = process_media(config, self.logger)
        
        assert stats["processed"] == 1
        
        # Find output file
        output_folder = self.output_dir / "Test" / "2021-06"
        files = list(output_folder.glob("*.jpg"))
        assert len(files) == 1
        
        filename = files[0].name
        # Should not have time
        assert "103045" not in filename
        
    def test_google_photos_structure(self):
        """Test processing a Google Photos-like structure."""
        fixtures = create_google_photos_structure(self.input_dir)
        for fixture in fixtures:
            fixture.create(self.input_dir)
            self.fixture_manager.fixtures.append(fixture)
        
        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
            copy_non_media=True,
        )
        
        stats = process_media(config, self.logger)
        
        # Check various files were processed correctly
        assert stats["errors"] == 0
        
        # Vacation album should exist
        vacation_folder = self.output_dir / "Test" / "2021-07"
        if vacation_folder.exists():
            files = list(vacation_folder.glob("*.jpg"))
            assert any("Vacation" in f.name for f in files)


class TestDuplicateDetection:
    """Integration tests for duplicate detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fixture_manager = FixtureManager()
        self.input_dir, self.output_dir = self.fixture_manager.setup()
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.DEBUG)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        self.fixture_manager.teardown()
        
    def test_same_file_different_locations(self):
        """Test that same file in different locations is detected as duplicate."""
        # Create same image in two locations
        from PIL import Image
        import shutil
        
        album1 = self.input_dir / "Album1"
        album1.mkdir()
        album2 = self.input_dir / "Album2"
        album2.mkdir()
        
        # Create original image
        img = Image.new("RGB", (100, 100), color="red")
        img.save(album1 / "photo.jpg")
        
        # Copy to second location
        shutil.copy(album1 / "photo.jpg", album2 / "photo.jpg")
        
        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
        )
        
        stats = process_media(config, self.logger)
        
        # One should be processed, one should be a duplicate
        assert stats["processed"] == 1
        assert stats["skipped_duplicate"] == 1


class TestDatabasePersistence:
    """Integration tests for database persistence."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fixture_manager = FixtureManager()
        self.input_dir, self.output_dir = self.fixture_manager.setup()
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.DEBUG)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        self.fixture_manager.teardown()
        
    def test_skip_already_processed(self):
        """Test that already-processed files are skipped on rerun."""
        fixture = ImageWithSidecar(
            name="test_photo",
            date_taken=datetime(2021, 6, 15, 10, 30, 0),
        )
        self.fixture_manager.add_fixture(fixture)
        
        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
        )
        
        # First run
        stats1 = process_media(config, self.logger)
        assert stats1["processed"] == 1
        assert stats1["skipped_known"] == 0
        
        # Second run
        stats2 = process_media(config, self.logger)
        assert stats2["processed"] == 0
        assert stats2["skipped_known"] == 1
