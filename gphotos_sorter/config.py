from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class StorageLayout(str, Enum):
    single = "single"
    year_month = "year/month"
    year_dash_month = "year-month"


class YearFormat(str, Enum):
    """Year format in filename."""
    YYYY = "YYYY"  # 2021
    YY = "YY"      # 21


class MonthFormat(str, Enum):
    """Month format in filename."""
    MM = "MM"           # 06
    name = "name"       # June
    short = "short"     # Jun


class DayFormat(str, Enum):
    """Day format in filename."""
    DD = "DD"           # 15
    weekday = "weekday" # 15_Tuesday


class DuplicatePolicy(str, Enum):
    """Policy for handling duplicate files with different resolutions."""
    keep_first = "keep-first"                      # Keep the first file encountered (current behavior)
    keep_higher_resolution = "keep-higher-resolution"  # Keep the file with higher resolution


class FilenameFormat(BaseModel):
    """Configurable filename format."""
    include_time: bool = Field(default=True, description="Include time (HHMMSS) in filename")
    year_format: YearFormat = Field(default=YearFormat.YYYY, description="Year format")
    month_format: MonthFormat = Field(default=MonthFormat.MM, description="Month format")
    day_format: DayFormat = Field(default=DayFormat.DD, description="Day format")
    include_tags: bool = Field(default=True, description="Include album tags in filename")
    max_tags: Optional[int] = Field(default=None, description="Maximum number of tags (None=no limit)")


class InputRoot(BaseModel):
    owner: str = Field(..., description="Owner label for the input root")
    path: Path = Field(..., description="Path to the input root")

    @field_validator("path")
    @classmethod
    def expand_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()


class AppConfig(BaseModel):
    """Configuration for media ingestion.
    
    All options can also be supplied via CLI flags. CLI flags override config file values.
    """
    input_roots: List[InputRoot] = Field(
        default_factory=list,
        description="List of input roots with owner and path"
    )
    output_root: Path = Field(
        ...,  # Required - no default
        description="Output root folder for organized media"
    )
    storage_layout: StorageLayout = Field(
        default=StorageLayout.year_dash_month,
        description="How to organize output folders: single, year/month, or year-month"
    )
    db_path: Optional[Path] = Field(
        default=None,
        description="Path to SQLite database (default: output_root/media.sqlite)"
    )
    filename_format: FilenameFormat = Field(
        default_factory=FilenameFormat,
        description="Filename format options"
    )
    copy_non_media: bool = Field(
        default=False,
        description="Copy non-media files to output (with warning)"
    )
    copy_sidecar: bool = Field(
        default=False,
        description="Copy sidecar JSON files alongside media"
    )
    modify_exif: bool = Field(
        default=True,
        description="Write metadata to EXIF tags"
    )
    dry_run: bool = Field(
        default=False,
        description="Don't actually copy files, just show what would be done"
    )
    duplicate_policy: DuplicatePolicy = Field(
        default=DuplicatePolicy.keep_first,
        description="Policy for handling duplicates with different resolutions"
    )

    @field_validator("output_root")
    @classmethod
    def expand_output(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("db_path")
    @classmethod
    def expand_db_path(cls, value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return None
        return value.expanduser().resolve()

    def resolve_db_path(self) -> Path:
        if self.db_path:
            return self.db_path
        return self.output_root / "media.sqlite"
