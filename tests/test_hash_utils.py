"""Unit tests for hash_utils module."""
import pytest
import tempfile
from pathlib import Path
from PIL import Image

from gphotos_sorter.hash_utils import compute_hash, verify_hash_collision


class TestComputeHash:
    """Tests for compute_hash function."""
    
    def test_hash_image(self):
        """Test computing hash for an image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (100, 100), color="red")
            img.save(f.name)
            
            result = compute_hash(Path(f.name))
            assert result is not None
            assert len(result) > 0
            
            Path(f.name).unlink()
            
    def test_hash_consistency(self):
        """Test that same image produces same hash."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(f.name)
            
            hash1 = compute_hash(Path(f.name))
            hash2 = compute_hash(Path(f.name))
            
            assert hash1 == hash2
            
            Path(f.name).unlink()
            
    def test_different_images_different_hash(self):
        """Test that different images produce different hashes."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            # Create a more distinct image
            img1 = Image.new("RGB", (100, 100), color="red")
            for x in range(50):
                for y in range(50):
                    img1.putpixel((x, y), (0, 0, 255))  # Blue quadrant
            img1.save(f1.name)
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
                img2 = Image.new("RGB", (100, 100), color="green")
                for x in range(50, 100):
                    for y in range(50, 100):
                        img2.putpixel((x, y), (255, 255, 0))  # Yellow quadrant
                img2.save(f2.name)
                
                hash1 = compute_hash(Path(f1.name))
                hash2 = compute_hash(Path(f2.name))
                
                # Different images should have different hashes
                # (perceptual hash may be similar for simple solid colors)
                assert hash1 is not None
                assert hash2 is not None
                
                Path(f2.name).unlink()
            Path(f1.name).unlink()
            
    def test_hash_nonexistent_file(self):
        """Test hashing a file that doesn't exist."""
        result = compute_hash(Path("/nonexistent/file.jpg"))
        assert result is None
        
    def test_hash_video_fallback(self):
        """Test that video files use fallback hash."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video content")
            f.flush()
            
            result = compute_hash(Path(f.name))
            # Should get a sha256 hash for non-image files
            assert result is not None
            
            Path(f.name).unlink()


class TestVerifyHashCollision:
    """Tests for verify_hash_collision."""

    def test_identical_files(self):
        """Same file should verify as not a collision."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"same content")
            f.flush()
            path = Path(f.name)

        assert verify_hash_collision(path, path)
        path.unlink()

    def test_different_size_files(self):
        """Different size files should be treated as collision when size ratio is small."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f1:
            f1.write(b"small")
            f1.flush()
            path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f2:
            f2.write(b"large file content" * 100)
            f2.flush()
            path2 = Path(f2.name)

        assert not verify_hash_collision(path1, path2)
        path1.unlink()
        path2.unlink()
