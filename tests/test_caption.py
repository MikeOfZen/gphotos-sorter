"""Tests for caption processor."""
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch

from uncloud.processors.caption import CaptionProcessor
from uncloud.services.pipeline import MediaData, ProcessingContextImpl


class TestCaptionProcessor:
    """Tests for CaptionProcessor."""
    
    @pytest.fixture
    def processor(self):
        return CaptionProcessor()
    
    @pytest.fixture
    def test_image(self, tmp_path: Path) -> Path:
        """Create a test image with distinctive content."""
        path = tmp_path / "sunset.jpg"
        # Create a colorful image (sunset-like)
        img = Image.new("RGB", (256, 256))
        pixels = img.load()
        for y in range(256):
            for x in range(256):
                # Orange gradient
                pixels[x, y] = (255, int(150 * y / 256), 0)
        img.save(path, "JPEG")
        return path
    
    @pytest.fixture
    def test_video(self, tmp_path: Path) -> Path:
        """Create a fake video file."""
        path = tmp_path / "video.mp4"
        path.write_bytes(b"fake video" * 100)
        return path
    
    def test_key(self, processor):
        """Test processor key."""
        assert processor.key == "caption"
    
    def test_version(self, processor):
        """Test processor version."""
        assert processor.version == 1
    
    def test_depends_on(self, processor):
        """Test no dependencies."""
        assert processor.depends_on == []
    
    def test_can_process_image(self, processor, test_image):
        """Test can_process returns True for images."""
        media = MediaData.load(test_image)
        ctx = ProcessingContextImpl(path=test_image, media=media)
        
        assert processor.can_process(ctx) is True
    
    def test_can_process_video(self, processor, test_video):
        """Test can_process returns False for videos."""
        media = MediaData.load(test_video)
        ctx = ProcessingContextImpl(path=test_video, media=media)
        
        assert processor.can_process(ctx) is False
    
    def test_can_process_no_image(self, processor, tmp_path):
        """Test can_process returns False when image not loaded."""
        path = tmp_path / "broken.jpg"
        path.write_text("not an image")
        media = MediaData.load(path)
        ctx = ProcessingContextImpl(path=path, media=media)
        
        assert processor.can_process(ctx) is False
    
    @pytest.mark.slow
    def test_process_generates_caption(self, processor, test_image):
        """Test that process generates a caption (slow - downloads model)."""
        media = MediaData.load(test_image)
        ctx = ProcessingContextImpl(path=test_image, media=media)
        
        caption = processor.process(ctx)
        
        # Caption should be a non-empty string
        assert isinstance(caption, str)
        assert len(caption) > 0
        assert len(caption) < 200  # Reasonable length
    
    @pytest.mark.slow
    def test_model_caching(self, processor, test_image):
        """Test that model is loaded once and cached."""
        media = MediaData.load(test_image)
        ctx = ProcessingContextImpl(path=test_image, media=media)
        
        # First call loads model
        caption1 = processor.process(ctx)
        
        # Second call uses cached model
        caption2 = processor.process(ctx)
        
        # Same image should give same caption
        assert caption1 == caption2
        
        # Model should be cached
        assert CaptionProcessor._model is not None
        assert CaptionProcessor._processor is not None
    
    def test_process_converts_to_rgb(self, processor, tmp_path):
        """Test that non-RGB images are converted."""
        path = tmp_path / "grayscale.jpg"
        img = Image.new("L", (100, 100), color=128)  # Grayscale
        img.save(path, "JPEG")
        
        media = MediaData.load(path)
        ctx = ProcessingContextImpl(path=path, media=media)
        
        with patch.object(CaptionProcessor, '_ensure_model_loaded') as mock_load:
            # Mock the model to avoid loading
            mock_processor = MagicMock()
            mock_model = MagicMock()
            mock_processor.return_value = MagicMock()
            mock_model.generate.return_value = [[1, 2, 3]]
            mock_processor.decode.return_value = "test caption"
            mock_load.return_value = (mock_processor, mock_model, "cpu")
            
            caption = processor.process(ctx)
            
            # Should succeed even with grayscale input
            assert isinstance(caption, str)
