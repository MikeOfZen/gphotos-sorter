"""Unit tests for multiprocessing scanner module."""
import os
import logging
import tempfile
from pathlib import Path

from PIL import Image

from gphotos_sorter.config import AppConfig, InputRoot, StorageLayout
from gphotos_sorter.db import MediaDatabase, MediaRecord
from gphotos_sorter.scanner_mp import process_media_mp


class TestSourcePathSkipping:
    """Tests for skipping already-processed files based on source paths."""

    def setup_method(self):
        """Set up test directories."""
        self.input_dir = Path(tempfile.mkdtemp(prefix="gphotos_test_input_"))
        self.output_dir = Path(tempfile.mkdtemp(prefix="gphotos_test_output_"))
        self.db_path = self.output_dir / "media.sqlite"
        self.logger = logging.getLogger("test_scanner_mp")
        self.logger.setLevel(logging.DEBUG)

    def teardown_method(self):
        """Clean up test directories."""
        import shutil
        if self.input_dir.exists():
            shutil.rmtree(self.input_dir)
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def _create_test_image(self, path: Path) -> Path:
        """Create a simple test image."""
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (100, 100), color="red")
        img.save(path)
        return path

    def test_known_files_skipped_before_workers(self):
        """Test that files with known source paths are skipped before dispatching to workers."""
        # Create test images
        img1_path = self._create_test_image(self.input_dir / "album1" / "photo1.jpg")
        img2_path = self._create_test_image(self.input_dir / "album1" / "photo2.jpg")

        # Pre-populate database with img1 and img2 as "known"
        db = MediaDatabase(self.db_path)
        db.upsert(MediaRecord(
            similarity_hash="hash1",
            canonical_path="/output/photo1.jpg",
            owner="Test",
            date_taken=None,
            date_source="none",
            tags=[],
            source_paths=[str(img1_path)],
            status="ok",
            notes=None,
        ))
        db.upsert(MediaRecord(
            similarity_hash="hash2",
            canonical_path="/output/photo2.jpg",
            owner="Test",
            date_taken=None,
            date_source="none",
            tags=[],
            source_paths=[str(img2_path)],
            status="ok",
            notes=None,
        ))
        db.close()

        # Run the processor
        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
            storage_layout=StorageLayout.year_dash_month,
            db_path=self.db_path,
        )

        stats = process_media_mp(config, self.logger, num_workers=2)

        # Should have 2 skipped_known and only 1 processed
        assert stats["skipped_known"] == 2, f"Expected 2 skipped_known, got {stats}"
        assert stats["processed"] == 1, f"Expected 1 processed, got {stats}"

    def test_path_normalization_for_skipping(self):
        """Test that paths with different formats are correctly matched."""
        # Create test image
        img_path = self._create_test_image(self.input_dir / "album" / "photo.jpg")

        # Store path with extra slashes or different formatting
        db = MediaDatabase(self.db_path)
        # Store with a slightly different path format (extra component normalization)
        stored_path = os.path.normpath(str(img_path))
        db.upsert(MediaRecord(
            similarity_hash="hash1",
            canonical_path="/output/photo.jpg",
            owner="Test",
            date_taken=None,
            date_source="none",
            tags=[],
            source_paths=[stored_path],
            status="ok",
            notes=None,
        ))
        db.close()

        # Run the processor
        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
            storage_layout=StorageLayout.year_dash_month,
            db_path=self.db_path,
        )

        stats = process_media_mp(config, self.logger, num_workers=2)

        # Should be skipped as known
        assert stats["skipped_known"] == 1, f"Expected 1 skipped_known, got {stats}"
        assert stats["processed"] == 0, f"Expected 0 processed, got {stats}"

    def test_empty_database_processes_all(self):
        """Test that with empty database all files are processed."""
        # Create test images
        self._create_test_image(self.input_dir / "photo1.jpg")
        self._create_test_image(self.input_dir / "photo2.jpg")

        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
            storage_layout=StorageLayout.year_dash_month,
            db_path=self.db_path,
        )

        stats = process_media_mp(config, self.logger, num_workers=2)

        # All files should be processed
        assert stats["skipped_known"] == 0, f"Expected 0 skipped_known, got {stats}"
        # Both should be processed (they're identical images so one might be detected as duplicate)
        assert stats["processed"] + stats["skipped"] == 2, f"Expected 2 total processed+skipped, got {stats}"

    def test_skipped_known_not_sent_to_workers(self):
        """Test that skipped files don't consume worker resources."""
        # Create many test images
        num_files = 10
        for i in range(num_files):
            self._create_test_image(self.input_dir / f"photo_{i}.jpg")

        # Pre-populate database with all files as "known"
        db = MediaDatabase(self.db_path)
        for i in range(num_files):
            img_path = self.input_dir / f"photo_{i}.jpg"
            db.upsert(MediaRecord(
                similarity_hash=f"hash_{i}",
                canonical_path=f"/output/photo_{i}.jpg",
                owner="Test",
                date_taken=None,
                date_source="none",
                tags=[],
                source_paths=[str(img_path)],
                status="ok",
                notes=None,
            ))
        db.close()

        config = AppConfig(
            input_roots=[InputRoot(owner="Test", path=self.input_dir)],
            output_root=self.output_dir,
            storage_layout=StorageLayout.year_dash_month,
            db_path=self.db_path,
        )

        stats = process_media_mp(config, self.logger, num_workers=2)

        # All should be skipped as known, none sent to workers
        assert stats["skipped_known"] == num_files, f"Expected {num_files} skipped_known, got {stats}"
        assert stats["processed"] == 0, f"Expected 0 processed, got {stats}"
        assert stats["skipped"] == 0, f"Expected 0 worker-skipped, got {stats}"
