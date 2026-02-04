"""File operations service."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.config import OutputLayout, FilenameFormat


MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

MONTH_SHORT = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}


class FileManager:
    """Handles file operations with crash recovery support."""
    
    def __init__(
        self, 
        output_root: Path, 
        layout: OutputLayout,
        dry_run: bool = False,
        owner_folder: Optional[str] = None,
    ):
        """Initialize file manager.
        
        Args:
            output_root: Root directory for output.
            layout: How to organize files.
            dry_run: If True, don't actually copy files.
            owner_folder: Optional top-level owner name.
        """
        self._output_root = output_root
        self._layout = layout
        self._dry_run = dry_run
        self._owner_folder = owner_folder
    
    def copy_file(self, source: Path, target: Path) -> bool:
        """Copy a file with metadata preservation.
        
        Args:
            source: Source file path.
            target: Target file path.
        
        Returns:
            True if successful.
        """
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            return True
        except Exception:
            return False
    
    def move_file(self, source: Path, target: Path) -> bool:
        """Move a file.
        
        Args:
            source: Source file path.
            target: Target file path.
        
        Returns:
            True if successful.
        """
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            return True
        except Exception:
            return False
    
    def delete_file(self, path: Path) -> bool:
        """Delete a file.
        
        Args:
            path: File to delete.
        
        Returns:
            True if successful.
        """
        try:
            path.unlink(missing_ok=True)
            return True
        except Exception:
            return False
    
    def ensure_directory(self, path: Path) -> None:
        """Ensure directory exists.
        
        Args:
            path: Directory to create.
        """
        path.mkdir(parents=True, exist_ok=True)
    
    def build_output_directory(self, date_taken: Optional[datetime]) -> Path:
        """Build output directory path based on layout and date.
        
        Args:
            date_taken: Date the media was taken.
        
        Returns:
            Full path to output directory.
        """
        # Start with owner folder if specified
        base = self._output_root / self._owner_folder if self._owner_folder else self._output_root
        
        if date_taken is None:
            return base / "unknown"
        
        year = str(date_taken.year)
        month = f"{date_taken.month:02d}"
        day = f"{date_taken.day:02d}"
        
        if self._layout == OutputLayout.YEAR_MONTH:
            return base / year / f"{year}-{month}"
        
        if self._layout == OutputLayout.YEAR_MONTH_DAY:
            return base / year / f"{year}-{month}" / f"{year}-{month}-{day}"
        
        if self._layout == OutputLayout.FLAT:
            return base
        
        if self._layout == OutputLayout.SINGLE:
            return base / "processed"
        
        return base / year / f"{year}-{month}"
    
    def build_filename(
        self,
        date_taken: Optional[datetime],
        tags: tuple[str, ...],
        extension: str,
        format_opts: FilenameFormat,
        counter: int = 0,
    ) -> str:
        """Build output filename.
        
        Args:
            date_taken: Date the media was taken.
            tags: Tags to include in filename.
            extension: File extension.
            format_opts: Filename format options.
            counter: Counter for uniqueness.
        
        Returns:
            Filename string.
        """
        parts = []
        
        # Date/time part
        if date_taken:
            timestamp = self._format_timestamp(date_taken, format_opts)
            parts.append(timestamp)
        
        # Tags part
        if format_opts.include_tags and tags:
            max_tags = format_opts.max_tags or len(tags)
            tag_part = "_".join(tags[:max_tags])
            if tag_part:
                parts.append(tag_part)
        
        # Build base
        if parts:
            base = "_".join(parts)
        else:
            base = "file"
        
        # Add counter if needed
        if counter > 0:
            base = f"{base}_{counter}"
        
        return f"{base}{extension.lower()}"
    
    def _format_timestamp(self, dt: datetime, fmt: FilenameFormat) -> str:
        """Format datetime for filename."""
        parts = []
        
        # Year
        if fmt.year_format == "YY":
            parts.append(f"{dt.year % 100:02d}")
        else:
            parts.append(str(dt.year))
        
        # Month
        if fmt.month_format == "MM":
            parts.append(f"{dt.month:02d}")
        elif fmt.month_format == "M":
            parts.append(str(dt.month))
        elif fmt.month_format == "MMM":
            parts.append(MONTH_SHORT.get(dt.month, ""))
        elif fmt.month_format == "MMMM":
            parts.append(MONTH_NAMES.get(dt.month, ""))
        
        # Day
        if fmt.day_format == "DD":
            parts.append(f"{dt.day:02d}")
        else:
            parts.append(str(dt.day))
        
        base = "-".join(parts)
        
        # Time
        if fmt.include_time:
            time_str = f"{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"
            return f"{base}_{time_str}"
        
        return base
    
    def find_unique_path(
        self,
        directory: Path,
        date_taken: Optional[datetime],
        tags: tuple[str, ...],
        extension: str,
        format_opts: FilenameFormat,
    ) -> Path:
        """Find a unique filename in directory.
        
        Args:
            directory: Target directory.
            date_taken: Date for filename.
            tags: Tags for filename.
            extension: File extension.
            format_opts: Filename format options.
        
        Returns:
            Full path to unique file.
        """
        self.ensure_directory(directory)
        
        for counter in range(1000):
            filename = self.build_filename(
                date_taken, tags, extension, format_opts, counter
            )
            candidate = directory / filename
            if not candidate.exists():
                return candidate
        
        # Fallback with timestamp
        import time
        fallback = f"{int(time.time())}{extension.lower()}"
        return directory / fallback
