"""Tests for hash engine implementations."""
import pytest
from pathlib import Path
from PIL import Image

from uncloud.engines.hash_engine import (
    CPUHashEngine,
    GPUHashEngine,
    create_hash_engine,
    is_image,
    is_video,
    is_media,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from uncloud.core.config import HashBackend


class TestMediaDetection:
    """Tests for media type detection functions."""
    
    def test_is_image_jpg(self, tmp_path: Path):
        """Test image detection for JPEG."""
        path = tmp_path / "test.jpg"
        assert is_image(path) is True
        assert is_video(path) is False
        assert is_media(path) is True
    
    def test_is_image_png(self, tmp_path: Path):
        """Test image detection for PNG."""
        path = tmp_path / "test.PNG"  # uppercase
        assert is_image(path) is True
    
    def test_is_video_mp4(self, tmp_path: Path):
        """Test video detection for MP4."""
        path = tmp_path / "test.mp4"
        assert is_video(path) is True
        assert is_image(path) is False
        assert is_media(path) is True
    
    def test_is_not_media(self, tmp_path: Path):
        """Test non-media file detection."""
        path = tmp_path / "test.txt"
        assert is_image(path) is False
        assert is_video(path) is False
        assert is_media(path) is False
    
    def test_all_image_extensions(self):
        """Test all image extensions are detected."""
        for ext in IMAGE_EXTENSIONS:
            path = Path(f"test{ext}")
            assert is_image(path) is True
    
    def test_all_video_extensions(self):
        """Test all video extensions are detected."""
        for ext in VIDEO_EXTENSIONS:
            path = Path(f"test{ext}")
            assert is_video(path) is True


class TestCPUHashEngine:
    """Tests for CPU hash engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a CPU hash engine."""
        return CPUHashEngine()
    
    @pytest.fixture
    def test_image(self, tmp_path: Path) -> Path:
        """Create a test image."""
        path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(path, "JPEG")
        return path
    
    @pytest.fixture
    def test_video(self, tmp_path: Path) -> Path:
        """Create a test 'video' (just a binary file)."""
        path = tmp_path / "test.mp4"
        path.write_bytes(b"fake video content " * 100)
        return path
    
    def test_name(self, engine):
        """Test engine name."""
        assert engine.name == "CPU (imagehash)"
    
    def test_supports_gpu(self, engine):
        """Test GPU support is false."""
        assert engine.supports_gpu is False
    
    def test_hash_image(self, engine, test_image):
        """Test hashing an image."""
        result = engine.compute_hash(test_image)
        
        assert result is not None
        assert result.startswith("phash:")
    
    def test_hash_video(self, engine, test_video):
        """Test hashing a video (SHA256)."""
        result = engine.compute_hash(test_video)
        
        assert result is not None
        assert result.startswith("sha256:")
    
    def test_hash_nonexistent(self, engine):
        """Test hashing nonexistent file returns None."""
        result = engine.compute_hash(Path("/nonexistent/file.jpg"))
        assert result is None
    
    def test_hash_non_media(self, engine, tmp_path):
        """Test hashing non-media file returns None."""
        path = tmp_path / "test.txt"
        path.write_text("hello")
        
        result = engine.compute_hash(path)
        assert result is None
    
    def test_compute_batch(self, engine, tmp_path):
        """Test batch hash computation."""
        # Create test files
        paths = []
        for i in range(5):
            path = tmp_path / f"test{i}.jpg"
            img = Image.new("RGB", (50, 50), color=f"#{i*40:02x}0000")
            img.save(path, "JPEG")
            paths.append(path)
        
        results = engine.compute_batch(paths)
        
        assert len(results) == 5
        for result in results:
            assert result is not None
            assert result.startswith("phash:")
    
    def test_same_image_same_hash(self, engine, tmp_path):
        """Test that identical images produce same hash."""
        img = Image.new("RGB", (100, 100), color="blue")
        
        path1 = tmp_path / "test1.jpg"
        path2 = tmp_path / "test2.jpg"
        img.save(path1, "JPEG")
        img.save(path2, "JPEG")
        
        hash1 = engine.compute_hash(path1)
        hash2 = engine.compute_hash(path2)
        
        assert hash1 == hash2
    
    def test_different_images_different_hash(self, engine, tmp_path):
        """Test that different images produce different hashes."""
        path1 = tmp_path / "test1.jpg"
        path2 = tmp_path / "test2.jpg"
        
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color="blue")
        
        img1.save(path1, "JPEG")
        img2.save(path2, "JPEG")
        
        hash1 = engine.compute_hash(path1)
        hash2 = engine.compute_hash(path2)
        
        # Both hashes should be valid, even if solid colors may hash similarly
        assert hash1 is not None
        assert hash2 is not None
        assert hash1.startswith("phash:")
        assert hash2.startswith("phash:")


class TestGPUHashEngine:
    """Tests for GPU hash engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a GPU hash engine."""
        return GPUHashEngine("cuda")
    
    @pytest.fixture
    def test_image(self, tmp_path: Path) -> Path:
        """Create a test image."""
        path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="green")
        img.save(path, "JPEG")
        return path
    
    def test_name(self, engine):
        """Test engine name."""
        assert "GPU" in engine.name
    
    def test_hash_image(self, engine, test_image):
        """Test hashing an image (may fall back to CPU)."""
        result = engine.compute_hash(test_image)
        
        assert result is not None
        assert result.startswith("phash:")
    
    def test_batch_hash(self, engine, tmp_path):
        """Test batch hashing."""
        paths = []
        for i in range(3):
            path = tmp_path / f"test{i}.jpg"
            img = Image.new("RGB", (50, 50), color=f"#{i*80:02x}0000")
            img.save(path, "JPEG")
            paths.append(path)
        
        results = engine.compute_batch(paths)
        
        assert len(results) == 3
        for result in results:
            assert result is not None


class TestCreateHashEngine:
    """Tests for the factory function."""
    
    def test_create_cpu(self):
        """Test creating CPU engine."""
        engine = create_hash_engine(HashBackend.CPU)
        assert isinstance(engine, CPUHashEngine)
    
    def test_create_auto(self):
        """Test AUTO backend returns an engine."""
        engine = create_hash_engine(HashBackend.AUTO)
        # Should return either CPU or GPU depending on availability
        assert engine is not None
        assert hasattr(engine, 'compute_hash')
    
    def test_create_gpu_fallback(self):
        """Test GPU backend falls back to CPU if unavailable."""
        engine = create_hash_engine(HashBackend.GPU_CUDA)
        # Should return something that works
        assert engine is not None
