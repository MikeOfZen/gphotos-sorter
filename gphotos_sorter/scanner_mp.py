"""Multiprocessing version of the scanner for faster processing."""
from __future__ import annotations

import logging
import multiprocessing as mp
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Optional

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
from .scanner import MEDIA_EXTENSIONS, sanitize_tag, MONTH_NAMES, MONTH_SHORT, WEEKDAY_NAMES


@dataclass
class WorkItem:
    """A file to be processed by a worker."""
    path: Path
    owner: str
    tags: list[str]
    sidecar_path: Optional[Path]
    is_media: bool = True  # False for non-media files


@dataclass
class WorkResult:
    """Result from a worker, ready for DB write."""
    record: MediaRecord
    action: str  # "insert", "update", "error", or "non_media"
    update_tags: Optional[list[str]] = None
    update_source: Optional[str] = None


# Sentinel to signal workers to stop
STOP_SENTINEL = "STOP"


def iter_work_items(root: Path, owner: str, recursive: bool = True, include_non_media: bool = False):
    """Generate work items from a directory."""
    iterator = root.rglob("*") if recursive else root.iterdir()
    for path in iterator:
        if not path.is_file():
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
        yield WorkItem(path=path, owner=owner, tags=tags, sidecar_path=sidecar, is_media=is_media)


def resolve_date(path: Path, sidecar_path: Optional[Path]) -> tuple[Optional[datetime], str]:
    """Resolve date from various sources."""
    exif_dt = extract_exif_datetime(path)
    if exif_dt:
        return exif_dt, "exif"
    if sidecar_path:
        sidecar = load_sidecar(sidecar_path)
        if sidecar:
            sidecar_dt = extract_sidecar_datetime(sidecar)
            if sidecar_dt:
                return sidecar_dt, "sidecar"
    for part in path.parts:
        parsed = parse_date_from_folder(part)
        if parsed:
            return parsed, "folder"
    return None, "missing"


def build_output_dir(base: Path, owner: str, taken: Optional[datetime], layout: StorageLayout) -> Path:
    """Build output directory based on layout."""
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
    
    If no timestamp, just tags (files go to 'unknown' folder instead).
    """
    if fmt is None:
        fmt = FilenameFormat()
    
    if taken:
        stamp = format_timestamp(taken, fmt)
    else:
        # No timestamp - just use tags, file will be in 'unknown' folder
        stamp = ""
    
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
    import time
    return output_dir / f"{int(time.time())}{ext.lower()}"


def worker_process(
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    config_dict: dict,
    worker_id: int,
):
    """Worker process that handles hashing, copying, and EXIF writing."""
    # Reconstruct config from dict
    output_root = Path(config_dict["output_root"])
    layout = StorageLayout(config_dict["storage_layout"])
    copy_non_media = config_dict.get("copy_non_media", True)
    
    # Reconstruct FilenameFormat
    fmt = FilenameFormat(
        include_time=config_dict.get("fmt_include_time", True),
        year_format=YearFormat(config_dict.get("fmt_year_format", "YYYY")),
        month_format=MonthFormat(config_dict.get("fmt_month_format", "MM")),
        day_format=DayFormat(config_dict.get("fmt_day_format", "DD")),
        include_tags=config_dict.get("fmt_include_tags", True),
        max_tags=config_dict.get("fmt_max_tags", 5),
    )
    
    while True:
        try:
            item = work_queue.get(timeout=1)
        except Empty:
            continue
        
        if item == STOP_SENTINEL:
            break
        
        work_item: WorkItem = item
        
        try:
            # Handle non-media files differently
            if not work_item.is_media:
                if copy_non_media:
                    non_media_dir = output_root / work_item.owner / "non_media"
                    non_media_dir.mkdir(parents=True, exist_ok=True)
                    dest = non_media_dir / work_item.path.name
                    counter = 0
                    while dest.exists():
                        counter += 1
                        dest = non_media_dir / f"{work_item.path.stem}_{counter}{work_item.path.suffix}"
                    shutil.copy2(work_item.path, dest)
                    # Create a minimal record for non-media
                    record = MediaRecord(
                        similarity_hash=f"non_media:{work_item.path}",
                        canonical_path=str(dest),
                        owner=work_item.owner,
                        date_taken=None,
                        date_source="none",
                        tags=[],
                        source_paths=[str(work_item.path)],
                        status="non_media",
                        notes=None,
                    )
                    result_queue.put(WorkResult(record=record, action="non_media"))
                continue
            
            # Compute hash for media files
            similarity_hash = compute_hash(work_item.path)
            if not similarity_hash:
                record = MediaRecord(
                    similarity_hash=f"error:{work_item.path}",
                    canonical_path="",
                    owner=work_item.owner,
                    date_taken=None,
                    date_source="error",
                    tags=work_item.tags,
                    source_paths=[str(work_item.path)],
                    status="error",
                    notes="hash_failed",
                )
                result_queue.put(WorkResult(record=record, action="error"))
                continue
            
            # Put hash result for duplicate check (will be handled by writer)
            # We'll send a "check" action first, then writer responds via a different mechanism
            # Actually, simpler: we compute hash and let writer decide insert/update
            
            date_taken, date_source = resolve_date(work_item.path, work_item.sidecar_path)
            output_dir = build_output_dir(output_root, work_item.owner, date_taken, layout)
            output_dir.mkdir(parents=True, exist_ok=True)
            canonical_path = find_unique_filename(output_dir, date_taken, work_item.tags, work_item.path.suffix, fmt)
            
            # Copy file
            shutil.copy2(work_item.path, canonical_path)
            
            # Process sidecar and write EXIF
            sidecar_extra = False
            if work_item.sidecar_path:
                sidecar = load_sidecar(work_item.sidecar_path)
                if sidecar:
                    try:
                        sidecar_to_exif(canonical_path, sidecar, work_item.tags)
                    except Exception:
                        pass
                    sidecar_extra = sidecar_has_extras(sidecar)
                    if sidecar_extra:
                        target_sidecar = canonical_path.with_suffix(
                            canonical_path.suffix + ".supplemental-metadata.json"
                        )
                        if not target_sidecar.exists():
                            shutil.copy2(work_item.sidecar_path, target_sidecar)
            elif work_item.tags:
                try:
                    sidecar_to_exif(canonical_path, {}, work_item.tags)
                except Exception:
                    pass
            
            status = "ok" if date_taken else "missing_date"
            record = MediaRecord(
                similarity_hash=similarity_hash,
                canonical_path=str(canonical_path),
                owner=work_item.owner,
                date_taken=date_taken.isoformat() if date_taken else None,
                date_source=date_source,
                tags=sorted(set(work_item.tags)),
                source_paths=[str(work_item.path)],
                status=status,
                notes=None if not sidecar_extra else "sidecar_copied",
            )
            result_queue.put(WorkResult(record=record, action="insert"))
            
        except Exception as e:
            record = MediaRecord(
                similarity_hash=f"error:{work_item.path}",
                canonical_path="",
                owner=work_item.owner,
                date_taken=None,
                date_source="error",
                tags=work_item.tags,
                source_paths=[str(work_item.path)],
                status="error",
                notes=str(e),
            )
            result_queue.put(WorkResult(record=record, action="error"))


def writer_process(
    result_queue: mp.Queue,
    stats_dict: dict,
    db_path: Path,
    total_items: int,
    stop_event: mp.Event,
):
    """Single writer process that handles all DB operations."""
    database = MediaDatabase(db_path)
    
    processed = 0
    skipped = 0
    errors = 0
    non_media = 0
    
    try:
        while not stop_event.is_set() or not result_queue.empty():
            try:
                result: WorkResult = result_queue.get(timeout=0.5)
            except Empty:
                continue
            
            if result.action == "error":
                errors += 1
                database.upsert(result.record)
            elif result.action == "non_media":
                non_media += 1
                # Don't add non-media to DB, just track count
            elif result.action == "insert":
                # Check if this hash already exists
                existing = database.get_by_hash(result.record.similarity_hash)
                if existing:
                    # Duplicate - update tags/sources, delete the copied file
                    database.update_existing(
                        result.record.similarity_hash,
                        result.record.tags,
                        result.record.source_paths,
                    )
                    # Remove the file we just copied (it's a duplicate)
                    try:
                        Path(result.record.canonical_path).unlink()
                    except Exception:
                        pass
                    skipped += 1
                else:
                    database.upsert(result.record)
                    processed += 1
            
            # Log progress periodically
            total = processed + skipped + errors + non_media
            if total % 100 == 0:
                print(f"Progress: {total}/{total_items} (processed={processed}, skipped={skipped}, non_media={non_media}, errors={errors})")
    finally:
        database.close()
        stats_dict["processed"] = processed
        stats_dict["skipped"] = skipped
        stats_dict["errors"] = errors
        stats_dict["non_media"] = non_media


def process_media_mp(
    config: AppConfig,
    logger: logging.Logger,
    limit: Optional[int] = None,
    recursive: bool = True,
    num_workers: int = 4,
) -> dict:
    """Process media files using multiprocessing."""
    db_path = config.resolve_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pre-load known source paths to skip already-processed files
    logger.info("Loading known source paths from database...")
    database = MediaDatabase(db_path)
    known_sources = database.get_all_source_paths()
    database.close()
    logger.info("Found %d known source paths", len(known_sources))
    
    # Get config options
    copy_non_media = config.copy_non_media
    fmt = config.filename_format or FilenameFormat()
    
    # Collect all work items, filtering out known sources
    logger.info("Scanning input directories...")
    work_items = []
    skipped_known = 0
    media_count = 0
    non_media_count = 0
    
    for input_root in config.input_roots:
        if not input_root.path.exists():
            logger.warning("Input root missing: %s", input_root.path)
            continue
        for item in iter_work_items(input_root.path, input_root.owner, recursive=recursive, include_non_media=copy_non_media):
            if str(item.path) in known_sources:
                skipped_known += 1
                continue  # Skip already processed
            work_items.append(item)
            if item.is_media:
                media_count += 1
            else:
                non_media_count += 1
                logger.warning("Non-media file: %s", item.path)
            if limit and len(work_items) >= limit:
                break
        if limit and len(work_items) >= limit:
            break
    
    total_items = len(work_items)
    logger.info("Found %d new files to process (media=%d, non_media=%d, skipped_known=%d)", 
                total_items, media_count, non_media_count, skipped_known)
    
    if total_items == 0:
        logger.info("============================================================")
        logger.info("PROCESSING SUMMARY")
        logger.info("============================================================")
        logger.info("No new files to process (all already in database)")
        logger.info("============================================================")
        return {"processed": 0, "skipped": 0, "errors": 0, "non_media": 0, "skipped_known": skipped_known}
    
    # Create queues
    work_queue = mp.Queue(maxsize=1000)
    result_queue = mp.Queue()
    stop_event = mp.Event()
    
    # Shared stats dict
    manager = mp.Manager()
    stats_dict = manager.dict()
    
    # Config dict for workers
    config_dict = {
        "output_root": str(config.output_root),
        "storage_layout": config.storage_layout.value,
        "copy_non_media": copy_non_media,
        "fmt_include_time": fmt.include_time,
        "fmt_year_format": fmt.year_format.value,
        "fmt_month_format": fmt.month_format.value,
        "fmt_day_format": fmt.day_format.value,
        "fmt_include_tags": fmt.include_tags,
        "fmt_max_tags": fmt.max_tags,
    }
    
    # Start worker processes
    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(work_queue, result_queue, config_dict, i),
        )
        p.start()
        workers.append(p)
    
    # Start writer process
    writer = mp.Process(
        target=writer_process,
        args=(result_queue, stats_dict, db_path, total_items, stop_event),
    )
    writer.start()
    
    # Feed work items
    logger.info("Starting processing with %d workers...", num_workers)
    for item in work_items:
        work_queue.put(item)
    
    # Send stop sentinels to workers
    for _ in range(num_workers):
        work_queue.put(STOP_SENTINEL)
    
    # Wait for workers to finish
    for p in workers:
        p.join()
    
    # Signal writer to stop and wait
    stop_event.set()
    writer.join()
    
    result = {
        "processed": stats_dict.get("processed", 0),
        "skipped": stats_dict.get("skipped", 0),
        "errors": stats_dict.get("errors", 0),
        "non_media": stats_dict.get("non_media", 0),
        "skipped_known": skipped_known,
    }
    
    # Print comprehensive summary
    logger.info("============================================================")
    logger.info("PROCESSING SUMMARY")
    logger.info("============================================================")
    logger.info("Total files discovered: %d", total_items + skipped_known)
    logger.info("  - Media files: %d", media_count + skipped_known)
    logger.info("  - Non-media files: %d", non_media_count)
    logger.info("Media file breakdown:")
    logger.info("  - Already known (skipped): %d", skipped_known)
    logger.info("  - New files processed: %d", result["processed"])
    logger.info("  - Duplicates found: %d", result["skipped"])
    logger.info("  - Errors: %d", result["errors"])
    if non_media_count > 0:
        logger.info("Non-media files copied: %d", result["non_media"])
    
    # Validation
    expected = media_count
    actual = result["processed"] + result["skipped"] + result["errors"]
    if expected == actual:
        logger.info("Validation: All files accounted for ✓")
    else:
        logger.warning("Validation MISMATCH: expected=%d, actual=%d", expected, actual)
    logger.info("============================================================")
    
    return result
