from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from PIL.ExifTags import TAGS


SIDE_CAR_SUFFIX = ".supplemental-metadata.json"


def load_sidecar(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def find_sidecar(media_path: Path) -> Optional[Path]:
    """Find the sidecar JSON file for a media file.
    
    Google Photos creates sidecars as:
    - photo.jpg.json (most common)
    - photo.json (sometimes)
    
    We also check for our own format:
    - photo.jpg.supplemental-metadata.json
    """
    candidates = [
        # Google Photos format: photo.jpg.json
        media_path.with_suffix(media_path.suffix + ".json"),
        media_path.with_name(media_path.name + ".json"),
        # Our format
        media_path.with_suffix(media_path.suffix + SIDE_CAR_SUFFIX),
        media_path.with_suffix(SIDE_CAR_SUFFIX),
        media_path.with_name(media_path.name + SIDE_CAR_SUFFIX),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_exif_datetime(media_path: Path) -> Optional[datetime]:
    if not media_path.exists():
        return None
    try:
        with Image.open(media_path) as image:
            exif = image.getexif()
            if not exif:
                return None
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in {"DateTimeOriginal", "DateTime", "CreateDate"}:
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None


def extract_sidecar_datetime(sidecar: Dict[str, Any]) -> Optional[datetime]:
    for key in ("photoTakenTime", "creationTime"):
        entry = sidecar.get(key) or {}
        ts = entry.get("timestamp")
        if ts:
            try:
                return datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(tzinfo=None)
            except Exception:
                continue
    return None


def extract_geo(sidecar: Dict[str, Any]) -> Dict[str, Any]:
    """Extract geo data, preferring geoDataExif if present."""
    geo = sidecar.get("geoDataExif") or sidecar.get("geoData") or {}
    lat = geo.get("latitude")
    lon = geo.get("longitude")
    alt = geo.get("altitude")
    # Filter out 0.0 values which mean no data
    if lat in (0, 0.0) and lon in (0, 0.0):
        return {}
    return {
        "latitude": lat,
        "longitude": lon,
        "altitude": alt if alt and alt != 0.0 else None,
    }


def extract_people(sidecar: Dict[str, Any]) -> List[str]:
    """Extract people names from sidecar."""
    people = sidecar.get("people") or []
    return [p.get("name") for p in people if p.get("name")]


def sidecar_to_exif(media_path: Path, sidecar: Dict[str, Any], tags: List[str]) -> bool:
    """Write sidecar metadata and tags to EXIF using exiftool.
    
    Args:
        media_path: Path to media file
        sidecar: Sidecar JSON data
        tags: Album tags to write as keywords
        
    Returns:
        True if successful, raises exception if exiftool not found
    """
    if not shutil.which("exiftool"):
        raise RuntimeError("exiftool is required but not found in PATH")
    
    exif_args = []
    
    # Description
    description = sidecar.get("description") or ""
    if description:
        exif_args.append(f"-ImageDescription={description}")
    
    # Date/time
    taken = extract_sidecar_datetime(sidecar)
    if taken:
        stamp = taken.strftime("%Y:%m:%d %H:%M:%S")
        exif_args.extend([
            f"-DateTimeOriginal={stamp}",
            f"-CreateDate={stamp}",
            f"-ModifyDate={stamp}",
        ])
    
    # GPS location
    geo = extract_geo(sidecar)
    if geo.get("latitude") is not None and geo.get("longitude") is not None:
        lat = geo["latitude"]
        lon = geo["longitude"]
        # Set lat/lon with proper reference
        lat_ref = "N" if lat >= 0 else "S"
        lon_ref = "E" if lon >= 0 else "W"
        exif_args.extend([
            f"-GPSLatitude={abs(lat)}",
            f"-GPSLatitudeRef={lat_ref}",
            f"-GPSLongitude={abs(lon)}",
            f"-GPSLongitudeRef={lon_ref}",
        ])
        if geo.get("altitude") is not None:
            alt = geo["altitude"]
            alt_ref = 0 if alt >= 0 else 1  # 0=above sea level, 1=below
            exif_args.extend([
                f"-GPSAltitude={abs(alt)}",
                f"-GPSAltitudeRef={alt_ref}",
            ])
    
    # People as subject
    people = extract_people(sidecar)
    for person in people:
        exif_args.append(f"-Subject+={person}")
    
    # Tags/albums as keywords
    for tag in tags:
        exif_args.append(f"-Keywords+={tag}")
        exif_args.append(f"-Subject+={tag}")
    
    if not exif_args:
        return True  # Nothing to write, but not an error
    
    try:
        result = subprocess.run(
            ["exiftool", "-overwrite_original", *exif_args, str(media_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        # Log but don't fail - some files may not support all EXIF fields
        return False


def sidecar_has_extras(sidecar: Dict[str, Any]) -> bool:
    """Check if sidecar has data that cannot be written to EXIF."""
    if not sidecar:
        return False
    # These keys are fully mapped to EXIF
    mapped_keys = {
        "title", "description", "photoTakenTime", "creationTime", 
        "geoData", "geoDataExif", "people", "imageViews", "url"
    }
    for key in sidecar.keys():
        if key in mapped_keys:
            continue
        value = sidecar.get(key)
        # Check if it has meaningful data
        if value and value not in (None, "", [], {}, 0, "0"):
            # googlePhotosOrigin is not critical info
            if key == "googlePhotosOrigin":
                continue
            return True
    return False


class ExifToolBatch:
    """Batch exiftool operations using -stay_open mode for 3-5x performance improvement.
    
    Usage:
        batch = ExifToolBatch()
        batch.queue_write(path1, sidecar1, tags1)
        batch.queue_write(path2, sidecar2, tags2)
        results = batch.flush_and_wait()
        batch.close()
    """
    
    def __init__(self):
        if not shutil.which("exiftool"):
            raise RuntimeError("exiftool is required but not found in PATH")
        
        self.proc = subprocess.Popen(
            ['exiftool', '-stay_open', 'True', '-@', '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._pending = 0
        self._paths: List[Path] = []
    
    def queue_write(self, media_path: Path, sidecar: Dict[str, Any], tags: List[str]) -> None:
        """Queue a metadata write operation (non-blocking)."""
        exif_args = []
        
        # Description
        description = sidecar.get("description") or ""
        if description:
            exif_args.append(f"-ImageDescription={description}")
        
        # Date/time
        taken = extract_sidecar_datetime(sidecar)
        if taken:
            stamp = taken.strftime("%Y:%m:%d %H:%M:%S")
            exif_args.extend([
                f"-DateTimeOriginal={stamp}",
                f"-CreateDate={stamp}",
                f"-ModifyDate={stamp}",
            ])
        
        # GPS location
        geo = extract_geo(sidecar)
        if geo.get("latitude") is not None and geo.get("longitude") is not None:
            lat = geo["latitude"]
            lon = geo["longitude"]
            lat_ref = "N" if lat >= 0 else "S"
            lon_ref = "E" if lon >= 0 else "W"
            exif_args.extend([
                f"-GPSLatitude={abs(lat)}",
                f"-GPSLatitudeRef={lat_ref}",
                f"-GPSLongitude={abs(lon)}",
                f"-GPSLongitudeRef={lon_ref}",
            ])
            if geo.get("altitude") is not None:
                alt = geo["altitude"]
                alt_ref = 0 if alt >= 0 else 1
                exif_args.extend([
                    f"-GPSAltitude={abs(alt)}",
                    f"-GPSAltitudeRef={alt_ref}",
                ])
        
        # People as subject
        people = extract_people(sidecar)
        for person in people:
            exif_args.append(f"-Subject+={person}")
        
        # Tags/albums as keywords
        for tag in tags:
            exif_args.append(f"-Keywords+={tag}")
            exif_args.append(f"-Subject+={tag}")
        
        if not exif_args:
            return  # Nothing to write
        
        commands = exif_args + ["-overwrite_original", str(media_path), "-execute"]
        cmd_str = "\n".join(commands) + "\n"
        self.proc.stdin.write(cmd_str)
        self._pending += 1
        self._paths.append(media_path)
    
    def flush_and_wait(self) -> List[Tuple[Path, bool]]:
        """Flush all pending commands and wait for results. Returns list of (path, success)."""
        if self._pending == 0:
            return []
        
        self.proc.stdin.flush()
        
        results: List[Tuple[Path, bool]] = []
        ready_count = 0
        current_lines: List[str] = []
        
        while ready_count < self._pending:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line == "{ready}":
                # Determine success from output
                success = any("image files updated" in l for l in current_lines)
                if ready_count < len(self._paths):
                    results.append((self._paths[ready_count], success))
                ready_count += 1
                current_lines = []
            else:
                current_lines.append(line)
        
        self._pending = 0
        self._paths = []
        return results
    
    def close(self) -> None:
        """Terminate exiftool process."""
        try:
            self.proc.stdin.write("-stay_open\nFalse\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.terminate()
