"""Unit tests for config module."""
import pytest
from pathlib import Path

from gphotos_sorter.config import (
    AppConfig,
    InputRoot,
    StorageLayout,
    FilenameFormat,
    YearFormat,
    MonthFormat,
    DayFormat,
)


class TestFilenameFormat:
    """Tests for FilenameFormat configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        fmt = FilenameFormat()
        assert fmt.include_time is True
        assert fmt.year_format == YearFormat.YYYY
        assert fmt.month_format == MonthFormat.MM
        assert fmt.day_format == DayFormat.DD
        assert fmt.include_tags is True
        assert fmt.max_tags is None  # No limit by default
        
    def test_custom_values(self):
        """Test custom configuration."""
        fmt = FilenameFormat(
            include_time=False,
            year_format=YearFormat.YY,
            month_format=MonthFormat.name,
            day_format=DayFormat.weekday,
            include_tags=True,
            max_tags=3,
        )
        assert fmt.include_time is False
        assert fmt.year_format == YearFormat.YY
        assert fmt.month_format == MonthFormat.name
        assert fmt.day_format == DayFormat.weekday
        assert fmt.max_tags == 3


class TestInputRoot:
    """Tests for InputRoot configuration."""
    
    def test_path_expansion(self):
        """Test that paths are expanded."""
        root = InputRoot(owner="Test", path=Path("~/photos"))
        assert not str(root.path).startswith("~")
        
    def test_owner_required(self):
        """Test that owner is required."""
        with pytest.raises(Exception):
            InputRoot(path=Path("/tmp"))


class TestAppConfig:
    """Tests for AppConfig configuration."""
    
    def test_output_root_required(self):
        """Test that output_root is required."""
        with pytest.raises(Exception):
            AppConfig()
            
    def test_default_layout(self):
        """Test default storage layout."""
        config = AppConfig(output_root=Path("/tmp/output"))
        assert config.storage_layout == StorageLayout.year_dash_month
        
    def test_resolve_db_path_default(self):
        """Test default database path resolution."""
        config = AppConfig(output_root=Path("/tmp/output"))
        assert config.resolve_db_path() == Path("/tmp/output/media.sqlite")
        
    def test_resolve_db_path_custom(self):
        """Test custom database path resolution."""
        config = AppConfig(
            output_root=Path("/tmp/output"),
            db_path=Path("/tmp/custom.sqlite"),
        )
        assert config.resolve_db_path() == Path("/tmp/custom.sqlite")
        
    def test_new_options(self):
        """Test new configuration options."""
        config = AppConfig(
            output_root=Path("/tmp/output"),
            copy_sidecar=True,
            modify_exif=False,
            dry_run=True,
        )
        assert config.copy_sidecar is True
        assert config.modify_exif is False
        assert config.dry_run is True
        
    def test_default_new_options(self):
        """Test default values for new options."""
        config = AppConfig(output_root=Path("/tmp/output"))
        assert config.copy_sidecar is False
        assert config.modify_exif is True
        assert config.dry_run is False
        assert config.copy_non_media is True


class TestStorageLayout:
    """Tests for StorageLayout enum."""
    
    def test_layout_values(self):
        """Test storage layout enum values."""
        assert StorageLayout.single.value == "single"
        assert StorageLayout.year_month.value == "year/month"
        assert StorageLayout.year_dash_month.value == "year-month"
