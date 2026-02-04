"""Tests for processing pipeline and processors."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from uncloud.services.pipeline import (
    MediaData,
    ProcessingContextImpl,
    MetadataService,
    ProcessingPipeline,
    PipelineStats,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from uncloud.processors.hash import PerceptualHashProcessor, VideoHashProcessor


class TestMediaData:
    """Tests for MediaData loading."""
    
    @pytest.fixture
    def test_image(self, tmp_path: Path) -> Path:
        """Create a test image."""
        path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(path, "JPEG")
        return path
    
    @pytest.fixture
    def test_video(self, tmp_path: Path) -> Path:
        """Create a fake video file."""
        path = tmp_path / "test.mp4"
        path.write_bytes(b"fake video content" * 1000)
        return path
    
    def test_load_image(self, test_image):
        """Test loading an image file."""
        media = MediaData.load(test_image)
        
        assert media.path == test_image
        assert media.image is not None
        assert media.is_video is False
        assert media._loaded is True
    
    def test_load_video(self, test_video):
        """Test loading a video file (no frame extraction)."""
        media = MediaData.load(test_video)
        
        assert media.path == test_video
        assert media.image is None  # Videos don't load image
        assert media.is_video is True
        assert media._loaded is True
    
    def test_load_nonexistent(self, tmp_path):
        """Test loading nonexistent file."""
        path = tmp_path / "nonexistent.jpg"
        media = MediaData.load(path)
        
        assert media.image is None
        assert media._loaded is False
    
    def test_load_unknown_extension(self, tmp_path):
        """Test loading file with unknown extension."""
        path = tmp_path / "file.xyz"
        path.write_text("content")
        media = MediaData.load(path)
        
        assert media.image is None
        assert media.is_video is False


class TestProcessingContextImpl:
    """Tests for ProcessingContextImpl."""
    
    @pytest.fixture
    def test_image(self, tmp_path: Path) -> Path:
        path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(path, "JPEG")
        return path
    
    def test_context_properties(self, test_image):
        """Test context property accessors."""
        media = MediaData.load(test_image)
        ctx = ProcessingContextImpl(
            path=test_image,
            media=media,
            results={'prev': 'result'},
            write_cache=True,
        )
        
        assert ctx.path == test_image
        assert ctx.image is not None
        assert ctx.is_video is False
        assert ctx.results == {'prev': 'result'}
        assert ctx.write_cache is True


class TestMetadataService:
    """Tests for MetadataService."""
    
    @pytest.fixture
    def mock_daemon_pool(self):
        """Create mock daemon pool."""
        pool = MagicMock()
        daemon = MagicMock()
        daemon.is_alive = True
        pool.get_daemon.return_value = daemon
        return pool, daemon
    
    def test_make_tag(self, mock_daemon_pool):
        """Test tag format creation."""
        pool, _ = mock_daemon_pool
        service = MetadataService(pool)
        
        tag = service._make_tag("phash", 1, "abc123")
        assert tag == 'uncloud:phash:1:"abc123"'
    
    def test_make_tag_complex_value(self, mock_daemon_pool):
        """Test tag with complex JSON value."""
        pool, _ = mock_daemon_pool
        service = MetadataService(pool)
        
        tag = service._make_tag("embeddings", 2, [1.0, 2.0, 3.0])
        assert tag == 'uncloud:embeddings:2:[1.0,2.0,3.0]'
    
    def test_parse_tag(self, mock_daemon_pool):
        """Test tag parsing."""
        pool, _ = mock_daemon_pool
        service = MetadataService(pool)
        
        result = service._parse_tag('uncloud:phash:1:"abc123"', "phash")
        assert result == (1, "abc123")
    
    def test_parse_tag_wrong_key(self, mock_daemon_pool):
        """Test parsing tag with wrong key returns None."""
        pool, _ = mock_daemon_pool
        service = MetadataService(pool)
        
        result = service._parse_tag('uncloud:phash:1:"abc123"', "clip")
        assert result is None
    
    def test_parse_tag_complex_value(self, mock_daemon_pool):
        """Test parsing complex value."""
        pool, _ = mock_daemon_pool
        service = MetadataService(pool)
        
        result = service._parse_tag('uncloud:faces:3:[{"x":10,"y":20}]', "faces")
        assert result == (3, [{"x": 10, "y": 20}])
    
    def test_read_cached_hit(self, mock_daemon_pool, tmp_path):
        """Test cache hit."""
        pool, daemon = mock_daemon_pool
        daemon.extract_subjects.return_value = ['uncloud:phash:1:"test_hash"']
        
        service = MetadataService(pool)
        result = service.read_cached(tmp_path / "test.jpg", "phash", min_version=1)
        
        assert result == "test_hash"
    
    def test_read_cached_stale(self, mock_daemon_pool, tmp_path):
        """Test cache miss due to old version."""
        pool, daemon = mock_daemon_pool
        daemon.extract_subjects.return_value = ['uncloud:phash:1:"old_hash"']
        
        service = MetadataService(pool)
        result = service.read_cached(tmp_path / "test.jpg", "phash", min_version=2)
        
        assert result is None  # Version 1 < 2
    
    def test_read_cached_miss(self, mock_daemon_pool, tmp_path):
        """Test cache miss (no data)."""
        pool, daemon = mock_daemon_pool
        daemon.extract_subjects.return_value = []
        
        service = MetadataService(pool)
        result = service.read_cached(tmp_path / "test.jpg", "phash", min_version=1)
        
        assert result is None
    
    def test_write_cached(self, mock_daemon_pool, tmp_path):
        """Test writing to cache."""
        pool, daemon = mock_daemon_pool
        daemon.write_subject.return_value = True
        
        service = MetadataService(pool)
        result = service.write_cached(tmp_path / "test.jpg", "phash", 1, "new_hash")
        
        assert result is True
        daemon.write_subject.assert_called_once()
        call_args = daemon.write_subject.call_args[0]
        assert 'uncloud:phash:1:"new_hash"' in call_args[1]


class TestPerceptualHashProcessor:
    """Tests for PerceptualHashProcessor."""
    
    @pytest.fixture
    def processor(self):
        return PerceptualHashProcessor()
    
    @pytest.fixture
    def test_image(self, tmp_path: Path) -> Path:
        path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="green")
        img.save(path, "JPEG")
        return path
    
    @pytest.fixture
    def test_video(self, tmp_path: Path) -> Path:
        path = tmp_path / "test.mp4"
        path.write_bytes(b"fake video content" * 1000)
        return path
    
    def test_key(self, processor):
        """Test processor key."""
        assert processor.key == "phash"
    
    def test_version(self, processor):
        """Test processor version."""
        assert processor.version == 1
    
    def test_depends_on(self, processor):
        """Test no dependencies."""
        assert processor.depends_on == []
    
    def test_can_process_always_true(self, processor, test_image):
        """Test can_process returns True."""
        media = MediaData.load(test_image)
        ctx = ProcessingContextImpl(path=test_image, media=media)
        
        assert processor.can_process(ctx) is True
    
    def test_process_image(self, processor, test_image):
        """Test processing an image."""
        media = MediaData.load(test_image)
        ctx = ProcessingContextImpl(path=test_image, media=media)
        
        result = processor.process(ctx)
        
        assert result.startswith("phash:")
        assert len(result) > 10  # Has actual hash content
    
    def test_process_video(self, processor, test_video):
        """Test processing a video."""
        media = MediaData.load(test_video)
        ctx = ProcessingContextImpl(path=test_video, media=media)
        
        result = processor.process(ctx)
        
        assert result.startswith("file:")
        assert len(result) > 10
    
    def test_same_image_same_hash(self, processor, test_image):
        """Test same image produces same hash."""
        media = MediaData.load(test_image)
        ctx = ProcessingContextImpl(path=test_image, media=media)
        
        result1 = processor.process(ctx)
        result2 = processor.process(ctx)
        
        assert result1 == result2


class TestVideoHashProcessor:
    """Tests for VideoHashProcessor."""
    
    @pytest.fixture
    def processor(self):
        return VideoHashProcessor()
    
    @pytest.fixture
    def test_image(self, tmp_path: Path) -> Path:
        path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="yellow")
        img.save(path, "JPEG")
        return path
    
    @pytest.fixture
    def test_video(self, tmp_path: Path) -> Path:
        path = tmp_path / "test.mp4"
        path.write_bytes(b"fake video content" * 1000)
        return path
    
    def test_key(self, processor):
        """Test processor key."""
        assert processor.key == "vhash"
    
    def test_can_process_video(self, processor, test_video):
        """Test can_process True for videos."""
        media = MediaData.load(test_video)
        ctx = ProcessingContextImpl(path=test_video, media=media)
        
        assert processor.can_process(ctx) is True
    
    def test_can_process_image(self, processor, test_image):
        """Test can_process False for images."""
        media = MediaData.load(test_image)
        ctx = ProcessingContextImpl(path=test_image, media=media)
        
        assert processor.can_process(ctx) is False


class TestProcessingPipeline:
    """Tests for ProcessingPipeline."""
    
    @pytest.fixture
    def mock_progress(self):
        """Create mock progress reporter."""
        progress = MagicMock()
        return progress
    
    @pytest.fixture
    def mock_metadata_service(self):
        """Create mock metadata service."""
        service = MagicMock()
        service.read_cached.return_value = None  # No cache by default
        service.write_cached.return_value = True
        return service
    
    @pytest.fixture
    def test_image(self, tmp_path: Path) -> Path:
        path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="purple")
        img.save(path, "JPEG")
        return path
    
    def test_pipeline_single_processor(self, mock_progress, mock_metadata_service, test_image):
        """Test pipeline with single processor."""
        processor = PerceptualHashProcessor()
        
        pipeline = ProcessingPipeline(
            processors=[processor],
            metadata_service=mock_metadata_service,
            progress=mock_progress,
            write_cache=True,
        )
        
        stats = PipelineStats(total_files=1)
        results = pipeline.process_file(test_image, stats)
        
        assert 'phash' in results
        assert results['phash'].startswith('phash:')
        assert stats.processor_stats['phash']['computed'] == 1
    
    def test_pipeline_cache_hit(self, mock_progress, mock_metadata_service, test_image):
        """Test pipeline with cache hit."""
        mock_metadata_service.read_cached.return_value = "phash:cached_value"
        processor = PerceptualHashProcessor()
        
        pipeline = ProcessingPipeline(
            processors=[processor],
            metadata_service=mock_metadata_service,
            progress=mock_progress,
        )
        
        stats = PipelineStats(total_files=1)
        results = pipeline.process_file(test_image, stats)
        
        assert results['phash'] == "phash:cached_value"
        assert stats.processor_stats['phash']['from_cache'] == 1
        assert stats.processor_stats['phash']['computed'] == 0
    
    def test_pipeline_write_cache_disabled(self, mock_progress, mock_metadata_service, test_image):
        """Test pipeline with cache writing disabled."""
        processor = PerceptualHashProcessor()
        
        pipeline = ProcessingPipeline(
            processors=[processor],
            metadata_service=mock_metadata_service,
            progress=mock_progress,
            write_cache=False,
        )
        
        stats = PipelineStats(total_files=1)
        pipeline.process_file(test_image, stats)
        
        # Should not write to cache
        mock_metadata_service.write_cached.assert_not_called()
    
    def test_pipeline_dependency_order(self, mock_progress, mock_metadata_service):
        """Test pipeline sorts by dependencies."""
        # Create mock processors with dependencies
        p1 = MagicMock()
        p1.key = "first"
        p1.version = 1
        p1.depends_on = []
        p1.can_process.return_value = True
        p1.process.return_value = "first_result"
        
        p2 = MagicMock()
        p2.key = "second"
        p2.version = 1
        p2.depends_on = ["first"]  # Depends on first
        p2.can_process.return_value = True
        p2.process.return_value = "second_result"
        
        # Pass in wrong order
        pipeline = ProcessingPipeline(
            processors=[p2, p1],  # Second before first
            metadata_service=mock_metadata_service,
            progress=mock_progress,
        )
        
        # Should be sorted correctly
        assert pipeline._processors[0].key == "first"
        assert pipeline._processors[1].key == "second"


class TestPipelineStats:
    """Tests for PipelineStats."""
    
    def test_init_processor(self):
        """Test initializing processor stats."""
        stats = PipelineStats()
        stats.init_processor("phash")
        
        assert "phash" in stats.processor_stats
        assert stats.processor_stats["phash"]["from_cache"] == 0
        assert stats.processor_stats["phash"]["computed"] == 0
    
    def test_record_cache_hit(self):
        """Test recording cache hit."""
        stats = PipelineStats()
        stats.init_processor("phash")
        stats.record_cache_hit("phash")
        
        assert stats.processor_stats["phash"]["from_cache"] == 1
    
    def test_record_computed(self):
        """Test recording computed result."""
        stats = PipelineStats()
        stats.init_processor("phash")
        stats.record_computed("phash", written=True)
        
        assert stats.processor_stats["phash"]["computed"] == 1
        assert stats.processor_stats["phash"]["written"] == 1


class TestExtensions:
    """Tests for extension constants."""
    
    def test_image_extensions(self):
        """Test image extensions set."""
        assert '.jpg' in IMAGE_EXTENSIONS
        assert '.jpeg' in IMAGE_EXTENSIONS
        assert '.png' in IMAGE_EXTENSIONS
        assert '.heic' in IMAGE_EXTENSIONS
    
    def test_video_extensions(self):
        """Test video extensions set."""
        assert '.mp4' in VIDEO_EXTENSIONS
        assert '.mov' in VIDEO_EXTENSIONS
        assert '.avi' in VIDEO_EXTENSIONS
        assert '.mkv' in VIDEO_EXTENSIONS
