"""Unit tests for scanner module."""
import pytest
from datetime import datetime
from pathlib import Path

from gphotos_sorter.scanner import (
    build_filename,
    build_output_dir,
    sanitize_tag,
    format_timestamp,
    MEDIA_EXTENSIONS,
)
from gphotos_sorter.config import (
    FilenameFormat,
    YearFormat,
    MonthFormat,
    DayFormat,
    StorageLayout,
)


class TestSanitizeTag:
    """Tests for sanitize_tag function."""
    
    def test_basic_tag(self):
        """Test basic tag sanitization."""
        assert sanitize_tag("Vacation") == "Vacation"
        assert sanitize_tag("Family Photos") == "Family_Photos"
        
    def test_special_characters(self):
        """Test removal of special characters."""
        # Characters are replaced with underscores, then collapsed
        result = sanitize_tag("Test<>:file")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        
        result = sanitize_tag("Path/to\\file")
        assert "/" not in result
        assert "\\" not in result
        
    def test_multiple_spaces(self):
        """Test collapsing of multiple spaces."""
        assert sanitize_tag("Test   Multiple   Spaces") == "Test_Multiple_Spaces"
        
    def test_length_limit(self):
        """Test tag length limiting."""
        long_tag = "A" * 50
        result = sanitize_tag(long_tag)
        assert len(result) <= 30
        
    def test_strip_underscores(self):
        """Test stripping of leading/trailing underscores."""
        assert sanitize_tag("_Test_") == "Test"


class TestFormatTimestamp:
    """Tests for format_timestamp function."""
    
    def setup_method(self):
        """Set up test date."""
        # Tuesday, June 15, 2021, 10:30:45
        self.dt = datetime(2021, 6, 15, 10, 30, 45)
        
    def test_default_format(self):
        """Test default timestamp format."""
        fmt = FilenameFormat()
        result = format_timestamp(self.dt, fmt)
        assert result == "20210615_103045"
        
    def test_yy_format(self):
        """Test YY year format."""
        fmt = FilenameFormat(year_format=YearFormat.YY)
        result = format_timestamp(self.dt, fmt)
        assert result.startswith("21")
        
    def test_month_name(self):
        """Test month name format."""
        fmt = FilenameFormat(month_format=MonthFormat.name)
        result = format_timestamp(self.dt, fmt)
        assert "June" in result
        
    def test_month_short(self):
        """Test short month format."""
        fmt = FilenameFormat(month_format=MonthFormat.short)
        result = format_timestamp(self.dt, fmt)
        assert "Jun" in result
        
    def test_weekday(self):
        """Test weekday format."""
        fmt = FilenameFormat(day_format=DayFormat.weekday)
        result = format_timestamp(self.dt, fmt)
        assert "Tuesday" in result
        
    def test_no_time(self):
        """Test format without time."""
        fmt = FilenameFormat(include_time=False)
        result = format_timestamp(self.dt, fmt)
        assert "103045" not in result
        assert result == "20210615"


class TestBuildFilename:
    """Tests for build_filename function."""
    
    def setup_method(self):
        """Set up test data."""
        self.dt = datetime(2021, 6, 15, 10, 30, 45)
        
    def test_basic_filename(self):
        """Test basic filename generation."""
        result = build_filename(self.dt, ["Album"], ".jpg", 0)
        assert result == "20210615_103045_Album.jpg"
        
    def test_multiple_tags(self):
        """Test filename with multiple tags."""
        result = build_filename(self.dt, ["Vacation", "Beach"], ".jpg", 0)
        assert "Vacation_Beach" in result
        
    def test_no_tags(self):
        """Test filename without tags."""
        fmt = FilenameFormat(include_tags=False)
        result = build_filename(self.dt, ["Album"], ".jpg", 0, fmt)
        assert "Album" not in result
        
    def test_max_tags_limit(self):
        """Test tag limit."""
        fmt = FilenameFormat(max_tags=2)
        result = build_filename(self.dt, ["A", "B", "C", "D"], ".jpg", 0, fmt)
        assert "A_B" in result
        assert "C" not in result
        
    def test_max_tags_none(self):
        """Test no tag limit."""
        fmt = FilenameFormat(max_tags=None)
        result = build_filename(self.dt, ["A", "B", "C", "D"], ".jpg", 0, fmt)
        assert "A_B_C_D" in result
        
    def test_collision_counter(self):
        """Test filename collision counter."""
        result = build_filename(self.dt, ["Album"], ".jpg", 3)
        assert "_3" in result
        
    def test_no_timestamp(self):
        """Test filename without timestamp."""
        result = build_filename(None, ["Album"], ".jpg", 0)
        assert result == "Album.jpg"
        
    def test_no_timestamp_no_tags(self):
        """Test filename without timestamp and tags."""
        result = build_filename(None, [], ".jpg", 0)
        assert result == "file.jpg"
        
    def test_extension_lowercase(self):
        """Test that extension is lowercased."""
        result = build_filename(self.dt, ["Album"], ".JPG", 0)
        assert result.endswith(".jpg")


class TestBuildOutputDir:
    """Tests for build_output_dir function."""
    
    def setup_method(self):
        """Set up test data."""
        self.base = Path("/output")
        self.dt = datetime(2021, 6, 15)
        
    def test_year_dash_month_layout(self):
        """Test year-month layout."""
        result = build_output_dir(self.base, "Owner", self.dt, StorageLayout.year_dash_month)
        assert result == Path("/output/Owner/2021-06")
        
    def test_year_month_layout(self):
        """Test year/month layout."""
        result = build_output_dir(self.base, "Owner", self.dt, StorageLayout.year_month)
        assert result == Path("/output/Owner/2021/06")
        
    def test_single_layout(self):
        """Test single folder layout."""
        result = build_output_dir(self.base, "Owner", self.dt, StorageLayout.single)
        assert result == Path("/output/Owner")
        
    def test_no_date(self):
        """Test output dir when no date is available."""
        result = build_output_dir(self.base, "Owner", None, StorageLayout.year_dash_month)
        assert result == Path("/output/Owner/unknown")


class TestMediaExtensions:
    """Tests for MEDIA_EXTENSIONS constant."""
    
    def test_common_image_formats(self):
        """Test that common image formats are included."""
        assert ".jpg" in MEDIA_EXTENSIONS
        assert ".jpeg" in MEDIA_EXTENSIONS
        assert ".png" in MEDIA_EXTENSIONS
        assert ".heic" in MEDIA_EXTENSIONS
        
    def test_common_video_formats(self):
        """Test that common video formats are included."""
        assert ".mp4" in MEDIA_EXTENSIONS
        assert ".mov" in MEDIA_EXTENSIONS
        assert ".mkv" in MEDIA_EXTENSIONS
        
    def test_lowercase(self):
        """Test that all extensions are lowercase."""
        for ext in MEDIA_EXTENSIONS:
            assert ext == ext.lower()
