"""Directory scanning service."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

from ..core.models import MediaItem
from ..core.config import InputSource
from ..engines.hash_engine import is_media


# Patterns for date-based folders
DATE_FOLDER_PATTERNS = [
    re.compile(r"^(19|20)\d{2}$"),  # YYYY
    re.compile(r"^(19|20)\d{2}[-_](0[1-9]|1[0-2])$"),  # YYYY-MM
    re.compile(r"^(19|20)\d{2}[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12]\d|3[01])$"),  # YYYY-MM-DD
]


def is_date_folder(name: str) -> bool:
    """Check if folder name looks like a date."""
    return any(p.match(name) for p in DATE_FOLDER_PATTERNS)


def find_sidecar(media_path: Path) -> Optional[Path]:
    """Find Google Photos JSON sidecar for a media file."""
    # Try exact match first: photo.jpg.json
    sidecar = media_path.with_suffix(media_path.suffix + ".json")
    if sidecar.exists():
        return sidecar
    
    # Try without extension: photo.json (for photo.jpg)
    sidecar = media_path.with_suffix(".json")
    if sidecar.exists():
        return sidecar
    
    return None


def extract_tags_from_path(path: Path, source_root: Path) -> tuple[str, ...]:
    """Extract album/folder tags from path relative to source."""
    try:
        rel_path = path.relative_to(source_root)
    except ValueError:
        return ()
    
    tags = []
    for part in rel_path.parts[:-1]:  # Exclude filename
        # Skip date-like folders
        if is_date_folder(part):
            continue
        # Skip common non-tag names
        if part.lower() in {"photos", "pictures", "images", "camera", "dcim"}:
            continue
        # Sanitize and add
        tag = sanitize_tag(part)
        if tag:
            tags.append(tag)
    
    return tuple(tags)


def sanitize_tag(tag: str) -> str:
    """Sanitize a tag string."""
    # Remove special characters
    tag = re.sub(r"[^\w\s-]", "", tag)
    # Collapse whitespace
    tag = re.sub(r"\s+", "_", tag.strip())
    # Limit length
    tag = tag[:50]
    # Strip trailing underscores
    return tag.strip("_")


class DirectoryScanner:
    """Scans directories for media files.
    
    Yields MediaItem objects for each discovered file.
    """
    
    def __init__(
        self,
        include_non_media: bool = False,
        follow_symlinks: bool = False,
    ):
        """Initialize the scanner.
        
        Args:
            include_non_media: Whether to include non-media files.
            follow_symlinks: Whether to follow symbolic links.
        """
        self._include_non_media = include_non_media
        self._follow_symlinks = follow_symlinks
    
    def scan(
        self,
        sources: list[InputSource],
        recursive: bool = True,
        skip_known: Optional[set[str]] = None,
    ) -> Iterator[MediaItem]:
        """Scan sources and yield MediaItems.
        
        Args:
            sources: List of input sources to scan.
            recursive: Whether to scan recursively.
            skip_known: Set of known paths to skip.
        
        Yields:
            MediaItem for each discovered file.
        """
        skip_known = skip_known or set()
        
        for source in sources:
            yield from self._scan_directory(
                source.path,
                source.owner,
                source.path,
                recursive,
                skip_known,
            )
    
    def _scan_directory(
        self,
        directory: Path,
        owner: str,
        source_root: Path,
        recursive: bool,
        skip_known: set[str],
    ) -> Iterator[MediaItem]:
        """Scan a single directory."""
        try:
            entries = list(directory.iterdir())
        except PermissionError:
            return
        
        for entry in entries:
            if entry.is_symlink() and not self._follow_symlinks:
                continue
            
            if entry.is_file():
                # Skip known files
                path_str = str(entry)
                if path_str in skip_known:
                    continue
                
                # Check if media
                if is_media(entry):
                    sidecar = find_sidecar(entry)
                    tags = extract_tags_from_path(entry, source_root)
                    yield MediaItem(
                        path=entry,
                        owner=owner,
                        tags=tags,
                        sidecar_path=sidecar,
                        is_media=True,
                    )
                elif self._include_non_media:
                    yield MediaItem(
                        path=entry,
                        owner=owner,
                        tags=extract_tags_from_path(entry, source_root),
                        is_media=False,
                    )
            
            elif entry.is_dir() and recursive:
                yield from self._scan_directory(
                    entry, owner, source_root, recursive, skip_known
                )
    
    def count_files(
        self,
        sources: list[InputSource],
        recursive: bool = True,
    ) -> tuple[int, int]:
        """Count files without loading them all.
        
        Returns:
            (media_count, non_media_count)
        """
        media = 0
        non_media = 0
        
        for item in self.scan(sources, recursive):
            if item.is_media:
                media += 1
            else:
                non_media += 1
        
        return media, non_media
