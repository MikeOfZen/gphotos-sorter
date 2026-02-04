"""Tests for core configuration."""
import pytest
from pathlib import Path

from gphotos_sorter.core.config import (
    OutputLayout,
    DuplicatePolicy,
    HashBackend,
    InputSource,
    FilenameFormat,
    SorterConfig,
)


class TestOutputLayout:
    """Tests for OutputLayout enum."""
    
    def test_values(self):
        """Test enum values."""
        assert OutputLayout.YEAR_MONTH.value == "year-month"
        assert OutputLayout.YEAR_MONTH_DAY.value == "year-month-day"
        assert OutputLayout.FLAT.value == "flat"
        assert OutputLayout.SINGLE.value == "single"


class TestDuplicatePolicy:
    """Tests for DuplicatePolicy enum."""
    
    def test_values(self):
        """Test enum values."""
        assert DuplicatePolicy.SKIP.value == "skip"
        assert DuplicatePolicy.KEEP_FIRST.value == "keep-first"
        assert DuplicatePolicy.KEEP_HIGHER_RESOLUTION.value == "keep-higher-resolution"


class TestHashBackend:
    """Tests for HashBackend enum."""
    
    def test_values(self):
        """Test enum values."""
        assert HashBackend.AUTO.value == "auto"
        assert HashBackend.CPU.value == "cpu"
        assert HashBackend.GPU_CUDA.value == "cuda"


class TestInputSource:
    """Tests for InputSource dataclass."""
    
    def test_create(self, tmp_path: Path):
        """Test basic creation."""
        source = InputSource(path=tmp_path, owner="alice")
        
        assert source.path == tmp_path
        assert source.owner == "alice"
    
    def test_default_owner(self, tmp_path: Path):
        """Test default owner."""
        source = InputSource(path=tmp_path)
        
        assert source.owner == "default"
    
    def test_nonexistent_path_raises(self):
        """Test that nonexistent path raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            InputSource(path=Path("/nonexistent/path"))
    
    def test_frozen(self, tmp_path: Path):
        """Test immutability."""
        source = InputSource(path=tmp_path, owner="alice")
        
        with pytest.raises(AttributeError):
            source.owner = "bob"
    
    def test_from_dict(self, tmp_path: Path):
        """Test from_dict factory."""
        source = InputSource.from_dict({
            "path": str(tmp_path),
            "owner": "charlie",
        })
        
        assert source.path == tmp_path
        assert source.owner == "charlie"


class TestFilenameFormat:
    """Tests for FilenameFormat dataclass."""
    
    def test_defaults(self):
        """Test default values."""
        fmt = FilenameFormat()
        
        assert fmt.include_time is True
        assert fmt.year_format == "YYYY"
        assert fmt.include_tags is True


class TestSorterConfig:
    """Tests for SorterConfig dataclass."""
    
    def test_create_minimal(self, tmp_path: Path):
        """Test minimal config creation."""
        output = tmp_path / "output"
        output.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        config = SorterConfig(
            output_root=output,
            inputs=(InputSource(path=input_dir, owner="alice"),),
        )
        
        assert config.output_root == output
        assert len(config.inputs) == 1
        assert config.layout == OutputLayout.YEAR_MONTH
    
    def test_db_path_default(self, tmp_path: Path):
        """Test db_path defaults to output_root/media.sqlite."""
        output = tmp_path / "output"
        output.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        config = SorterConfig(
            output_root=output,
            inputs=(InputSource(path=input_dir),),
        )
        
        assert config.resolved_db_path == output / "media.sqlite"
    
    def test_validation_no_inputs(self, tmp_path: Path):
        """Test validation fails with no inputs."""
        output = tmp_path / "output"
        output.mkdir()
        
        with pytest.raises(ValueError, match="At least one input source"):
            SorterConfig(output_root=output, inputs=())
    
    def test_validation_workers(self, tmp_path: Path):
        """Test validation fails with invalid workers."""
        output = tmp_path / "output"
        output.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        with pytest.raises(ValueError, match="Workers must be at least 1"):
            SorterConfig(
                output_root=output,
                inputs=(InputSource(path=input_dir),),
                workers=0,
            )
    
    def test_custom_values(self, tmp_path: Path):
        """Test custom configuration values."""
        output = tmp_path / "output"
        output.mkdir()
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        config = SorterConfig(
            output_root=output,
            inputs=(InputSource(path=input_dir),),
            layout=OutputLayout.FLAT,
            duplicate_policy=DuplicatePolicy.SKIP,
            workers=8,
            dry_run=True,
        )
        
        assert config.layout == OutputLayout.FLAT
        assert config.duplicate_policy == DuplicatePolicy.SKIP
        assert config.workers == 8
        assert config.dry_run is True
