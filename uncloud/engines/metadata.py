"""Metadata extraction and writing implementations."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from PIL import Image

from .hash_engine import IMAGE_EXTENSIONS


# Custom XMP namespace for uncloud metadata
# We use XMP:Description field with custom prefix for compatibility
UNCLOUD_HASH_TAG = "XMP-dc:Description"  # Store in Description temporarily
UNCLOUD_HASH_PREFIX = "uncloud:hash:"
UNCLOUD_TAGS_PREFIX = "uncloud:tags:"


class ExifToolMetadataExtractor:
    """Metadata extractor using exiftool for writing and PIL for reading.
    
    Uses exiftool's -stay_open mode for batch operations.
    
    Stores custom uncloud metadata (hash, tags) in XMP fields so the
    file becomes the source of truth and the database is just an index.
    """
    
    def __init__(self, use_batch_mode: bool = True):
        """Initialize the extractor.
        
        Args:
            use_batch_mode: Use exiftool's -stay_open for batch writes.
        """
        self._use_batch = use_batch_mode
        self._process: Optional[subprocess.Popen] = None
        self._pending_commands: list[str] = []
        self._batch_size = 20
        
        if use_batch_mode:
            self._start_exiftool()
    
    def _start_exiftool(self) -> None:
        """Start exiftool in stay_open mode."""
        try:
            self._process = subprocess.Popen(
                ["exiftool", "-stay_open", "True", "-@", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            self._process = None
            self._use_batch = False
    
    def extract_uncloud_hash(self, path: Path) -> Optional[str]:
        """Extract the stored uncloud hash from file metadata.
        
        This allows fast re-indexing without re-computing hashes.
        
        Args:
            path: Path to the file.
            
        Returns:
            The stored hash, or None if not found.
        """
        try:
            result = subprocess.run(
                ["exiftool", "-XMP:Subject", "-s", "-s", "-s", str(path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Subject may contain multiple values, look for our hash
                subjects = result.stdout.strip().split(", ")
                for subj in subjects:
                    if subj.startswith(UNCLOUD_HASH_PREFIX):
                        return subj[len(UNCLOUD_HASH_PREFIX):]
            return None
        except Exception:
            return None
    
    def write_uncloud_hash(self, path: Path, hash_value: str) -> bool:
        """Write the uncloud hash to file metadata.
        
        Stores in XMP:Subject as 'uncloud:hash:HASHVALUE' to avoid
        conflicting with other metadata.
        
        Args:
            path: Path to the file.
            hash_value: The similarity hash to store.
            
        Returns:
            True if successful.
        """
        tag_value = f"{UNCLOUD_HASH_PREFIX}{hash_value}"
        return self.write_tags(path, {"XMP:Subject+": tag_value})
    
    def extract_uncloud_metadata(self, path: Path) -> dict[str, Any]:
        """Extract all uncloud-specific metadata from file.
        
        Args:
            path: Path to the file.
            
        Returns:
            Dict with keys: 'hash', 'tags' (may be None).
        """
        result = {
            "hash": None,
            "tags": [],
        }
        
        try:
            proc = subprocess.run(
                ["exiftool", "-XMP:Subject", "-s", "-s", "-s", str(path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                subjects = proc.stdout.strip().split(", ")
                for subj in subjects:
                    if subj.startswith(UNCLOUD_HASH_PREFIX):
                        result["hash"] = subj[len(UNCLOUD_HASH_PREFIX):]
                    elif subj.startswith(UNCLOUD_TAGS_PREFIX):
                        tags_str = subj[len(UNCLOUD_TAGS_PREFIX):]
                        result["tags"] = tags_str.split("|") if tags_str else []
        except Exception:
            pass
        
        return result
    
    def write_uncloud_metadata(
        self,
        path: Path,
        hash_value: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> bool:
        """Write uncloud metadata to file.
        
        Args:
            path: Path to the file.
            hash_value: Optional hash to store.
            tags: Optional list of tags.
            
        Returns:
            True if successful.
        """
        subjects = []
        if hash_value:
            subjects.append(f"{UNCLOUD_HASH_PREFIX}{hash_value}")
        if tags:
            subjects.append(f"{UNCLOUD_TAGS_PREFIX}{'|'.join(tags)}")
        
        if not subjects:
            return True
        
        # Write each subject as a separate tag
        tag_dict = {}
        for i, subj in enumerate(subjects):
            tag_dict[f"XMP:Subject+"] = subj
            
        return self.write_tags(path, tag_dict)
    
    def extract_datetime(
        self, 
        path: Path, 
        sidecar: Optional[Path] = None
    ) -> tuple[Optional[datetime], str]:
        """Extract date taken from file or sidecar.
        
        Priority:
        1. Sidecar JSON (Google Photos)
        2. EXIF DateTimeOriginal
        3. EXIF CreateDate
        4. File modification time
        
        Returns:
            (datetime, source_description)
        """
        # Try sidecar first
        if sidecar and sidecar.exists():
            dt = self._extract_from_sidecar(sidecar)
            if dt:
                return dt, "sidecar"
        
        # Try EXIF
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            dt = self._extract_from_exif(path)
            if dt:
                return dt, "exif"
        
        # Fall back to file modification time
        try:
            mtime = path.stat().st_mtime
            return datetime.fromtimestamp(mtime), "file_mtime"
        except Exception:
            return None, "unknown"
    
    def _extract_from_sidecar(self, sidecar: Path) -> Optional[datetime]:
        """Extract datetime from Google Photos JSON sidecar."""
        try:
            with sidecar.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Google Photos format
            if "photoTakenTime" in data:
                timestamp = int(data["photoTakenTime"]["timestamp"])
                return datetime.fromtimestamp(timestamp)
            
            # Alternative format
            if "creationTime" in data:
                timestamp = int(data["creationTime"]["timestamp"])
                return datetime.fromtimestamp(timestamp)
            
            return None
        except Exception:
            return None
    
    def _extract_from_exif(self, path: Path) -> Optional[datetime]:
        """Extract datetime from EXIF data."""
        try:
            with Image.open(path) as img:
                exif = img._getexif()
                if not exif:
                    return None
                
                # Look for DateTimeOriginal (36867) or DateTimeDigitized (36868)
                for tag_id in (36867, 36868, 306):  # DateTimeOriginal, DateTimeDigitized, DateTime
                    if tag_id in exif:
                        dt_str = exif[tag_id]
                        return self._parse_exif_datetime(dt_str)
                
                return None
        except Exception:
            return None
    
    @staticmethod
    def _parse_exif_datetime(dt_str: str) -> Optional[datetime]:
        """Parse EXIF datetime string."""
        formats = [
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        return None
    
    def extract_resolution(self, path: Path) -> tuple[Optional[int], Optional[int]]:
        """Extract image dimensions."""
        try:
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                with Image.open(path) as img:
                    return img.size  # (width, height)
            return None, None
        except Exception:
            return None, None
    
    def write_tags(self, path: Path, tags: dict[str, Any]) -> bool:
        """Write EXIF tags to file.
        
        Args:
            path: File to write to.
            tags: Dict of tag_name -> value.
        
        Returns:
            True if successful.
        """
        if not tags:
            return True
        
        if self._use_batch and self._process:
            return self._batch_write(path, tags)
        return self._single_write(path, tags)
    
    def _batch_write(self, path: Path, tags: dict[str, Any]) -> bool:
        """Queue a write operation for batch processing."""
        if not self._process:
            return self._single_write(path, tags)
        
        # Build command
        cmd_parts = ["-overwrite_original"]
        for tag, value in tags.items():
            cmd_parts.append(f"-{tag}={value}")
        cmd_parts.append(str(path))
        cmd_parts.append("-execute")
        
        self._pending_commands.extend(cmd_parts)
        
        # Flush if batch is full
        if len(self._pending_commands) >= self._batch_size * 10:
            return self.flush()
        
        return True
    
    def _single_write(self, path: Path, tags: dict[str, Any]) -> bool:
        """Write tags using a single exiftool invocation."""
        try:
            args = ["exiftool", "-overwrite_original"]
            for tag, value in tags.items():
                args.append(f"-{tag}={value}")
            args.append(str(path))
            
            result = subprocess.run(args, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def flush(self) -> bool:
        """Flush all pending batch operations."""
        if not self._process or not self._pending_commands:
            return True
        
        try:
            # Send all commands
            for cmd in self._pending_commands:
                self._process.stdin.write(cmd + "\n")
            self._process.stdin.flush()
            
            # Read responses
            for _ in range(self._pending_commands.count("-execute")):
                while True:
                    line = self._process.stdout.readline()
                    if "{ready}" in line:
                        break
            
            self._pending_commands.clear()
            return True
        except Exception:
            return False
    
    def close(self) -> None:
        """Clean up resources."""
        if self._process is None:
            return
            
        try:
            # Try graceful shutdown first
            if self._process.poll() is None:  # Process still running
                try:
                    self._process.stdin.write("-stay_open\nFalse\n")
                    self._process.stdin.flush()
                    self._process.wait(timeout=2)
                except (BrokenPipeError, OSError):
                    pass  # Process already dead
        except Exception:
            pass
        finally:
            # Force kill if still alive
            try:
                if self._process.poll() is None:
                    self._process.kill()
                    self._process.wait(timeout=1)
            except Exception:
                pass
            self._process = None
    
    def __enter__(self) -> "ExifToolMetadataExtractor":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
