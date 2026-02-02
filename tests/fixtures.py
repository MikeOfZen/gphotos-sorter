"""Test fixtures for integration tests.

This module provides fixture classes that generate test data,
write it to disk, and know their expected output values.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image


@dataclass
class MediaFixture(ABC):
    """Base class for test media fixtures.
    
    Each fixture knows:
    - How to create its source file(s)
    - What the expected output path should be
    - What metadata should be in the output
    """
    name: str
    owner: str = "TestOwner"
    parent_folder: Optional[str] = None  # Album name from folder
    
    @abstractmethod
    def create(self, base_path: Path) -> Path:
        """Create the fixture file(s) and return the main file path."""
        pass
    
    @abstractmethod
    def expected_output_folder(self) -> str:
        """Return the expected output folder name (e.g., '2021-06' or 'unknown')."""
        pass
    
    @abstractmethod
    def expected_filename_contains(self) -> list[str]:
        """Return strings that should appear in the output filename."""
        pass


@dataclass
class ImageWithExifDate(MediaFixture):
    """Image with date embedded in EXIF."""
    date_taken: datetime = field(default_factory=lambda: datetime(2021, 6, 15, 10, 30, 45))
    
    def create(self, base_path: Path) -> Path:
        folder = base_path / (self.parent_folder or "")
        folder.mkdir(parents=True, exist_ok=True)
        
        file_path = folder / f"{self.name}.jpg"
        
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        img.save(file_path)
        
        # Note: To properly test EXIF, we'd need to write EXIF data
        # For now, we rely on sidecar or folder name for date
        
        return file_path
    
    def expected_output_folder(self) -> str:
        return f"{self.date_taken.year:04d}-{self.date_taken.month:02d}"
    
    def expected_filename_contains(self) -> list[str]:
        parts = [f"{self.date_taken.year:04d}{self.date_taken.month:02d}{self.date_taken.day:02d}"]
        if self.parent_folder:
            parts.append(self.parent_folder)
        return parts


@dataclass
class ImageWithSidecar(MediaFixture):
    """Image with Google Photos sidecar JSON."""
    date_taken: datetime = field(default_factory=lambda: datetime(2021, 6, 15, 10, 30, 45))
    description: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    def create(self, base_path: Path) -> Path:
        folder = base_path / (self.parent_folder or "")
        folder.mkdir(parents=True, exist_ok=True)
        
        file_path = folder / f"{self.name}.jpg"
        sidecar_path = folder / f"{self.name}.jpg.json"
        
        # Create image
        img = Image.new("RGB", (100, 100), color="green")
        img.save(file_path)
        
        # Create sidecar
        sidecar = {
            "title": f"{self.name}.jpg",
            "photoTakenTime": {
                "timestamp": str(int(self.date_taken.timestamp())),
                "formatted": self.date_taken.strftime("%b %d, %Y, %I:%M:%S %p UTC"),
            },
        }
        if self.description:
            sidecar["description"] = self.description
        if self.latitude is not None and self.longitude is not None:
            sidecar["geoData"] = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "altitude": 0.0,
            }
        
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)
        
        return file_path
    
    def expected_output_folder(self) -> str:
        return f"{self.date_taken.year:04d}-{self.date_taken.month:02d}"
    
    def expected_filename_contains(self) -> list[str]:
        parts = [f"{self.date_taken.year:04d}{self.date_taken.month:02d}{self.date_taken.day:02d}"]
        if self.parent_folder:
            parts.append(self.parent_folder)
        return parts


@dataclass  
class ImageWithDateFolder(MediaFixture):
    """Image in a date-named folder (no EXIF or sidecar)."""
    folder_date: str = "2021-06"  # The folder name
    
    def create(self, base_path: Path) -> Path:
        album_folder = self.parent_folder or ""
        if album_folder:
            folder = base_path / album_folder / self.folder_date
        else:
            folder = base_path / self.folder_date
        folder.mkdir(parents=True, exist_ok=True)
        
        file_path = folder / f"{self.name}.jpg"
        
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(file_path)
        
        return file_path
    
    def expected_output_folder(self) -> str:
        return self.folder_date
    
    def expected_filename_contains(self) -> list[str]:
        # Extract year and month from folder_date
        parts = self.folder_date.split("-")
        if len(parts) == 2:
            return [f"{parts[0]}{parts[1]}"]
        return []


@dataclass
class ImageNoDate(MediaFixture):
    """Image with no date information anywhere."""
    
    def create(self, base_path: Path) -> Path:
        folder = base_path / (self.parent_folder or "")
        folder.mkdir(parents=True, exist_ok=True)
        
        file_path = folder / f"{self.name}.jpg"
        
        img = Image.new("RGB", (100, 100), color="yellow")
        img.save(file_path)
        
        return file_path
    
    def expected_output_folder(self) -> str:
        return "unknown"
    
    def expected_filename_contains(self) -> list[str]:
        # Should just have album name
        if self.parent_folder:
            return [self.parent_folder]
        return []


@dataclass
class NonMediaFile(MediaFixture):
    """Non-media file (e.g., JSON, TXT)."""
    extension: str = ".json"
    content: str = '{"test": true}'
    
    def create(self, base_path: Path) -> Path:
        folder = base_path / (self.parent_folder or "")
        folder.mkdir(parents=True, exist_ok=True)
        
        file_path = folder / f"{self.name}{self.extension}"
        
        with open(file_path, "w") as f:
            f.write(self.content)
        
        return file_path
    
    def expected_output_folder(self) -> str:
        return "non_media"
    
    def expected_filename_contains(self) -> list[str]:
        return [self.name]


class FixtureManager:
    """Manages test fixtures - creates and cleans up."""
    
    def __init__(self):
        self.input_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.fixtures: list[MediaFixture] = []
        self.created_files: list[Path] = []
        
    def setup(self) -> tuple[Path, Path]:
        """Create temporary directories and return (input_dir, output_dir)."""
        self.input_dir = Path(tempfile.mkdtemp(prefix="gphotos_test_input_"))
        self.output_dir = Path(tempfile.mkdtemp(prefix="gphotos_test_output_"))
        return self.input_dir, self.output_dir
    
    def add_fixture(self, fixture: MediaFixture) -> Path:
        """Add a fixture and create its file(s)."""
        if self.input_dir is None:
            raise RuntimeError("Must call setup() before adding fixtures")
        
        file_path = fixture.create(self.input_dir)
        self.fixtures.append(fixture)
        self.created_files.append(file_path)
        return file_path
    
    def teardown(self):
        """Clean up all temporary files and directories."""
        if self.input_dir and self.input_dir.exists():
            shutil.rmtree(self.input_dir)
        if self.output_dir and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.fixtures = []
        self.created_files = []


def create_google_photos_structure(base_path: Path) -> list[MediaFixture]:
    """Create a Google Photos-like takeout structure with various test cases."""
    fixtures = []
    
    # Album with sidecar dates
    fixtures.append(ImageWithSidecar(
        name="IMG_001",
        parent_folder="Vacation 2021",
        date_taken=datetime(2021, 7, 15, 14, 30, 0),
        description="Beach sunset",
        latitude=36.7783,
        longitude=-119.4179,
    ))
    
    # Album without sidecar
    fixtures.append(ImageNoDate(
        name="facebook_photo",
        parent_folder="Facebook Uploads",
    ))
    
    # Date folder structure
    fixtures.append(ImageWithDateFolder(
        name="photo_in_date_folder",
        folder_date="2020-12",
    ))
    
    # Photos from YYYY folder
    fixtures.append(ImageWithSidecar(
        name="IMG_002",
        parent_folder="Photos from 2019",
        date_taken=datetime(2019, 3, 10, 9, 0, 0),
    ))
    
    # Non-media file
    fixtures.append(NonMediaFile(
        name="metadata",
        extension=".json",
        content='{"version": 1}',
    ))
    
    return fixtures
