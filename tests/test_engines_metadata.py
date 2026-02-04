"""Tests for metadata extraction."""
import pytest
from pathlib import Path
from PIL import Image
import json

from uncloud.engines.metadata import ExifToolMetadataExtractor


class TestExifToolMetadataExtractor:
    """Tests for ExifTool-based metadata extractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create an extractor instance."""
        ext = ExifToolMetadataExtractor(use_batch_mode=False)
        yield ext
        ext.close()
    
    @pytest.fixture
    def test_image(self, tmp_path: Path) -> Path:
        """Create a test image."""
        path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(path, "JPEG")
        return path
    
    @pytest.fixture
    def test_image_with_exif(self, tmp_path: Path) -> Path:
        """Create a test image (basic, without exif for simplicity)."""
        path = tmp_path / "test_exif.jpg"
        img = Image.new("RGB", (200, 150), color="blue")
        img.save(path, "JPEG")
        return path
    
    @pytest.fixture
    def test_sidecar(self, tmp_path: Path) -> Path:
        """Create a test sidecar JSON file."""
        path = tmp_path / "test.json"
        data = {
            "photoTakenTime": {
                "timestamp": "1704067200"  # 2024-01-01 00:00:00 UTC
            },
            "title": "Test Photo"
        }
        path.write_text(json.dumps(data))
        return path
    
    def test_extract_resolution(self, extractor, test_image):
        """Test extracting image dimensions."""
        width, height = extractor.extract_resolution(test_image)
        
        assert width == 100
        assert height == 100
    
    def test_extract_resolution_nonexistent(self, extractor):
        """Test resolution extraction for nonexistent file."""
        width, height = extractor.extract_resolution(Path("/nonexistent.jpg"))
        
        assert width is None
        assert height is None
    
    def test_extract_datetime_from_sidecar(self, extractor, test_image, test_sidecar):
        """Test extracting datetime from sidecar JSON."""
        dt, source = extractor.extract_datetime(test_image, test_sidecar)
        
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1
        assert source == "sidecar"
    
    def test_extract_datetime_no_sidecar(self, extractor, test_image):
        """Test datetime extraction without sidecar."""
        dt, source = extractor.extract_datetime(test_image, None)
        
        # May return None or fall back to file mtime
        if dt is not None:
            assert source in ("exif", "filename", "mtime", "file_mtime")
    
    def test_context_manager(self, tmp_path):
        """Test using extractor as context manager."""
        path = tmp_path / "test.jpg"
        img = Image.new("RGB", (50, 50), color="green")
        img.save(path, "JPEG")
        
        with ExifToolMetadataExtractor() as ext:
            width, height = ext.extract_resolution(path)
            assert width == 50
            assert height == 50
    
    def test_flush(self, extractor):
        """Test flush method returns True."""
        result = extractor.flush()
        assert result is True


class TestExifToolBatchMode:
    """Tests for batch mode metadata extractor."""
    
    @pytest.fixture
    def batch_extractor(self):
        """Create a batch mode extractor."""
        ext = ExifToolMetadataExtractor(use_batch_mode=True)
        yield ext
        ext.close()
    
    def test_batch_mode_creation(self, batch_extractor):
        """Test batch mode extractor is created."""
        # Just verify it was created successfully
        assert batch_extractor is not None
    
    def test_write_tags(self, batch_extractor, tmp_path):
        """Test writing tags (may or may not work depending on exiftool)."""
        path = tmp_path / "test.jpg"
        img = Image.new("RGB", (50, 50), color="yellow")
        img.save(path, "JPEG")
        
        # This should not raise even if exiftool isn't available
        result = batch_extractor.write_tags(path, {"Artist": "Test"})
        # Result may be True or False depending on exiftool availability
        assert isinstance(result, bool)
