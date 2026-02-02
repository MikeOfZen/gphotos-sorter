from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .config import AppConfig, StorageLayout, FilenameFormat, YearFormat, MonthFormat, DayFormat
from .date_utils import is_date_folder, parse_date_from_folder
from .db import MediaDatabase, MediaRecord
from .hash_utils import compute_hash
from .metadata_utils import (
    extract_exif_datetime,
    extract_sidecar_datetime,
    find_sidecar,
    load_sidecar,
    sidecar_has_extras,
    sidecar_to_exif,
)

MEDIA_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
    ".heif",
    ".tif",
    ".tiff",
    ".bmp",
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
}


def sanitize_tag(tag: str) -> str:
    """Sanitize a tag for use in filename."""
    # Remove/replace characters that are problematic in filenames
    tag = re.sub(r'[<>:"/\\|?*]', '_', tag)
    tag = re.sub(r'\s+', '_', tag)
    tag = re.sub(r'_+', '_', tag)
    return tag.strip('_')[:30]  # Limit length


@dataclass
class DiscoveredMedia:
    path: Path
    owner: str
    tags: list[str]
    sidecar_path: Optional[Path]
    is_media: bool = True  # False for non-media files


def iter_media_files(root: Path, owner: str, recursive: bool = True, include_non_media: bool = False) -> Iterable[DiscoveredMedia]:
    """Iterate over files in root directory.
    
    Args:
        root: Root directory to scan
        owner: Owner label for files
        recursive: Whether to scan subdirectories
        include_non_media: If True, also yield non-media files with is_media=False
    """
    iterator = root.rglob("*") if recursive else root.iterdir()
    for path in iterator:
        if not path.is_file():
            continue
        # Skip JSON sidecars
        if path.suffix.lower() == ".json":
            continue
        is_media = path.suffix.lower() in MEDIA_EXTENSIONS
        if not is_media and not include_non_media:
            continue
        relative_parts = path.relative_to(root).parts
        tags = []
        for idx, part in enumerate(relative_parts[:-1]):
            parent = relative_parts[idx - 1] if idx > 0 else None
            if not is_date_folder(part, parent=parent):
                tags.append(part)
        sidecar = find_sidecar(path) if is_media else None
        yield DiscoveredMedia(path=path, owner=owner, tags=tags, sidecar_path=sidecar, is_media=is_media)


def resolve_date(media: DiscoveredMedia) -> tuple[Optional[datetime], str]:
    exif_dt = extract_exif_datetime(media.path)
    if exif_dt:
        return exif_dt, "exif"
    if media.sidecar_path:
        sidecar = load_sidecar(media.sidecar_path)
        if sidecar:
            sidecar_dt = extract_sidecar_datetime(sidecar)
            if sidecar_dt:
                return sidecar_dt, "sidecar"
    for part in media.path.parts:
        parsed = parse_date_from_folder(part)
        if parsed:
            return parsed, "folder"
    return None, "missing"


MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
MONTH_SHORT = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def build_output_dir(base: Path, owner: str, taken: Optional[datetime], layout: StorageLayout) -> Path:
    if not taken:
        return base / owner / "unknown"
    if layout == StorageLayout.single:
        return base / owner
    if layout == StorageLayout.year_month:
        return base / owner / f"{taken.year:04d}" / f"{taken.month:02d}"
    if layout == StorageLayout.year_dash_month:
        return base / owner / f"{taken.year:04d}-{taken.month:02d}"
    return base / owner


def format_timestamp(taken: datetime, fmt: FilenameFormat) -> str:
    """Format timestamp according to FilenameFormat options."""
    parts = []
    
    # Year
    if fmt.year_format == YearFormat.YYYY:
        parts.append(f"{taken.year:04d}")
    else:  # YY
        parts.append(f"{taken.year % 100:02d}")
    
    # Month
    if fmt.month_format == MonthFormat.MM:
        parts.append(f"{taken.month:02d}")
    elif fmt.month_format == MonthFormat.name:
        parts.append(MONTH_NAMES[taken.month])
    else:  # short
        parts.append(MONTH_SHORT[taken.month])
    
    # Day
    if fmt.day_format == DayFormat.DD:
        parts.append(f"{taken.day:02d}")
    else:  # weekday
        parts.append(f"{taken.day:02d}_{WEEKDAY_NAMES[taken.weekday()]}")
    
    date_part = "".join(parts[:3]) if fmt.month_format == MonthFormat.MM else "_".join(parts[:3])
    
    # Time (optional)
    if fmt.include_time:
        time_part = taken.strftime("%H%M%S")
        return f"{date_part}_{time_part}"
    return date_part


def build_filename(taken: Optional[datetime], tags: list[str], ext: str, counter: int = 0, 
                   fmt: Optional[FilenameFormat] = None) -> str:
    """Build filename as timestamp_tags.ext.
    
    Format is configurable via FilenameFormat:
    - Year: YYYY or YY
    - Month: MM, full name, or short name
    - Day: DD or DD_weekday
    - Time: HHMMSS (optional)
    - Tags: album names (optional, configurable max)
    
    If no timestamp, just tags (files go to 'unknown' folder instead).
    If collision, append _N counter.
    """
    if fmt is None:
        fmt = FilenameFormat()
    
    if taken:
        stamp = format_timestamp(taken, fmt)
    else:
        # No timestamp - just use tags, file will be in 'unknown' folder
        stamp = ""
    
    # Add sanitized tags
    tag_part = ""
    if fmt.include_tags and tags:
        # max_tags None means no limit
        tags_to_use = tags if fmt.max_tags is None else tags[:fmt.max_tags]
        sanitized = [sanitize_tag(t) for t in tags_to_use if sanitize_tag(t)]
        if sanitized:
            tag_part = "_".join(sanitized)
    
    # Build base filename
    if stamp and tag_part:
        base = f"{stamp}_{tag_part}"
    elif stamp:
        base = stamp
    elif tag_part:
        base = tag_part
    else:
        # No timestamp and no tags - use counter or extension only
        base = "file"
    
    if counter > 0:
        base = f"{base}_{counter}"
    
    return f"{base}{ext.lower()}"


def find_unique_filename(output_dir: Path, taken: Optional[datetime], tags: list[str], ext: str,
                         fmt: Optional[FilenameFormat] = None) -> Path:
    """Find a unique filename, adding counter if needed."""
    for counter in range(1000):
        filename = build_filename(taken, tags, ext, counter, fmt)
        candidate = output_dir / filename
        if not candidate.exists():
            return candidate
    # Fallback with timestamp
    import time
    return output_dir / f"{int(time.time())}{ext.lower()}"


def process_media(config: AppConfig, logger: logging.Logger, limit: Optional[int] = None, recursive: bool = True) -> dict:
    """Process media files and return stats."""
    db_path = config.resolve_db_path()
    
    # In dry-run mode, don't create directories or database
    if config.dry_run:
        logger.info("DRY RUN MODE - no files will be copied or modified")
        database = None
        known_sources = set()
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        database = MediaDatabase(db_path)
        # Pre-load known source paths to skip already-processed files
        logger.info("Loading known source paths from database...")
        known_sources = database.get_all_source_paths()
        logger.info("Found %d known source paths", len(known_sources))
    
    # Stats counters
    stats = {
        "total_discovered": 0,
        "media_files": 0,
        "non_media_files": 0,
        "processed": 0,
        "skipped_duplicate": 0,
        "skipped_known": 0,
        "non_media_copied": 0,
        "errors": 0,
        "error_details": [],
    }

    try:
        for input_root in config.input_roots:
            if not input_root.path.exists():
                logger.warning("Input root missing: %s", input_root.path)
                stats["errors"] += 1
                stats["error_details"].append(f"Missing input root: {input_root.path}")
                continue
            
            for media in iter_media_files(input_root.path, input_root.owner, recursive=recursive,
                                          include_non_media=config.copy_non_media):
                stats["total_discovered"] += 1
                
                if limit and stats["processed"] >= limit:
                    logger.info("Reached limit of %d files", limit)
                    break
                
                # Handle non-media files
                if not media.is_media:
                    stats["non_media_files"] += 1
                    logger.warning("Non-media file: %s", media.path)
                    # Copy to non_media folder
                    if not config.dry_run:
                        try:
                            output_dir = config.output_root / media.owner / "non_media"
                            output_dir.mkdir(parents=True, exist_ok=True)
                            dest = output_dir / media.path.name
                            if not dest.exists():
                                shutil.copy2(media.path, dest)
                                stats["non_media_copied"] += 1
                                logger.info("Copied non-media: %s -> %s", media.path, dest)
                        except Exception as e:
                            logger.error("Failed to copy non-media file %s: %s", media.path, e)
                            stats["errors"] += 1
                            stats["error_details"].append(f"Non-media copy failed: {media.path}: {e}")
                    else:
                        logger.debug("Would copy non-media: %s", media.path)
                        stats["non_media_copied"] += 1
                    continue
                
                stats["media_files"] += 1
                
                # Skip files we've already processed
                source_str = str(media.path)
                if source_str in known_sources:
                    stats["skipped_known"] += 1
                    continue
                
                try:
                    logger.info("Processing %s", media.path)
                    
                    # In dry-run mode, skip most of the work
                    if config.dry_run:
                        date_taken, date_source = resolve_date(media)
                        output_dir = build_output_dir(config.output_root, media.owner, date_taken, config.storage_layout)
                        canonical_path = find_unique_filename(output_dir, date_taken, media.tags, media.path.suffix,
                                                              config.filename_format)
                        logger.debug("Would copy: %s -> %s", media.path, canonical_path)
                        if media.sidecar_path and config.copy_sidecar:
                            logger.debug("Would copy sidecar: %s", media.sidecar_path)
                        stats["processed"] += 1
                        continue
                    
                    similarity_hash = compute_hash(media.path)
                    if not similarity_hash:
                        logger.error("Hash failed: %s", media.path)
                        record = MediaRecord(
                            similarity_hash=f"error:{media.path}",
                            canonical_path="",
                            owner=media.owner,
                            date_taken=None,
                            date_source="error",
                            tags=media.tags,
                            source_paths=[str(media.path)],
                            status="error",
                            notes="hash_failed",
                        )
                        database.upsert(record)
                        stats["errors"] += 1
                        stats["error_details"].append(f"Hash failed: {media.path}")
                        continue

                    existing = database.get_by_hash(similarity_hash)
                    if existing:
                        database.update_existing(similarity_hash, media.tags, [str(media.path)])
                        logger.info("Duplicate found, updated tags/sources: %s", media.path)
                        stats["skipped_duplicate"] += 1
                        continue

                    date_taken, date_source = resolve_date(media)
                    output_dir = build_output_dir(config.output_root, media.owner, date_taken, config.storage_layout)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    canonical_path = find_unique_filename(output_dir, date_taken, media.tags, media.path.suffix,
                                                          config.filename_format)

                    # Copy file
                    shutil.copy2(media.path, canonical_path)
                    
                    # Process sidecar and write to EXIF
                    sidecar_extra = False
                    if media.sidecar_path:
                        sidecar = load_sidecar(media.sidecar_path)
                        if sidecar:
                            # Write sidecar metadata to EXIF (if enabled)
                            if config.modify_exif:
                                try:
                                    sidecar_to_exif(canonical_path, sidecar, media.tags)
                                except Exception as e:
                                    logger.warning("Failed to write EXIF for %s: %s", canonical_path, e)
                            
                            sidecar_extra = sidecar_has_extras(sidecar)
                            # Copy sidecar if requested or if it has extras
                            if config.copy_sidecar or sidecar_extra:
                                target_sidecar = canonical_path.with_suffix(canonical_path.suffix + ".supplemental-metadata.json")
                                if not target_sidecar.exists():
                                    shutil.copy2(media.sidecar_path, target_sidecar)
                    elif media.tags and config.modify_exif:
                        # No sidecar but we have tags - write tags to EXIF
                        try:
                            sidecar_to_exif(canonical_path, {}, media.tags)
                        except Exception as e:
                            logger.warning("Failed to write tags to EXIF for %s: %s", canonical_path, e)

                    status = "ok" if date_taken else "missing_date"
                    record = MediaRecord(
                        similarity_hash=similarity_hash,
                        canonical_path=str(canonical_path),
                        owner=media.owner,
                        date_taken=date_taken.isoformat() if date_taken else None,
                        date_source=date_source,
                        tags=sorted(set(media.tags)),
                        source_paths=[str(media.path)],
                        status=status,
                        notes=None if not sidecar_extra else "sidecar_copied",
                    )
                    database.upsert(record)
                    stats["processed"] += 1
                    
                except Exception as e:
                    logger.error("Failed to process %s: %s", media.path, e)
                    stats["errors"] += 1
                    stats["error_details"].append(f"Processing failed: {media.path}: {e}")
            
            if limit and stats["processed"] >= limit:
                break
        
        # Validation: compare counts
        expected_processed = stats["media_files"] - stats["skipped_known"]
        actual_accounted = stats["processed"] + stats["skipped_duplicate"] + stats["errors"]
        
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info("Total files discovered: %d", stats["total_discovered"])
        logger.info("  - Media files: %d", stats["media_files"])
        logger.info("  - Non-media files: %d", stats["non_media_files"])
        logger.info("Media file breakdown:")
        logger.info("  - Already known (skipped): %d", stats["skipped_known"])
        logger.info("  - New files processed: %d", stats["processed"])
        logger.info("  - Duplicates found: %d", stats["skipped_duplicate"])
        logger.info("  - Errors: %d", stats["errors"])
        if stats["non_media_copied"] > 0:
            logger.info("Non-media files copied: %d", stats["non_media_copied"])
        
        # Validation check
        if expected_processed != actual_accounted:
            logger.warning("COUNT MISMATCH! Expected %d new files, accounted for %d",
                          expected_processed, actual_accounted)
            logger.warning("Missing: %d files", expected_processed - actual_accounted)
        else:
            logger.info("Validation: All files accounted for ✓")
        
        if stats["error_details"]:
            logger.info("Error details:")
            for err in stats["error_details"][:10]:  # Limit to first 10
                logger.info("  - %s", err)
            if len(stats["error_details"]) > 10:
                logger.info("  ... and %d more errors", len(stats["error_details"]) - 10)
        
        logger.info("=" * 60)
        
        return stats
    finally:
        if database:
            database.close()
