"""Multiprocessing version of the scanner for faster processing."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Full
from typing import Optional

from .config import AppConfig, StorageLayout, FilenameFormat, YearFormat, MonthFormat, DayFormat, DuplicatePolicy
from .date_utils import is_date_folder, parse_date_from_folder
from .db import MediaDatabase, MediaRecord
from .hash_utils import compute_hash, get_image_resolution, sha256_file, verify_hash_collision
from .metadata_utils import (
    extract_exif_datetime,
    extract_sidecar_datetime,
    find_sidecar,
    load_sidecar,
    sidecar_has_extras,
    sidecar_to_exif,
    ExifToolBatch,
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
class HashResult:
    """Result from hashing/metadata phase (no copy yet)."""
    path: Path
    owner: str
    tags: list[str]
    sidecar_path: Optional[Path]
    is_media: bool
    similarity_hash: Optional[str] = None
    date_taken: Optional[datetime] = None
    date_source: str = "missing"
    width: Optional[int] = None
    height: Optional[int] = None
    error: Optional[str] = None


@dataclass
class CopyPlan:
    """Plan for copying a media file after dedupe decisions."""
    source_path: Path
    owner: str
    filename_tags: list[str]
    record_tags: list[str]
    record_source_paths: list[str]
    sidecar_path: Optional[Path]
    date_taken: Optional[datetime]
    date_source: str
    similarity_hash: str
    width: Optional[int]
    height: Optional[int]
    status: str
    canonical_path: Path
    replace_existing_path: Optional[Path] = None


@dataclass
class CopyResult:
    """Result from copy phase, ready for DB write."""
    record: MediaRecord
    pending_op_id: Optional[int]
    replace_existing_path: Optional[Path]
    action: str  # "insert", "error", or "non_media"
    error: Optional[str] = None


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
    """Worker process that handles hashing and metadata extraction (no copying)."""
    try:
        while True:
            try:
                item = work_queue.get(timeout=1)
            except Empty:
                continue

            if item == STOP_SENTINEL:
                break

            work_item: WorkItem = item

            try:
                if not work_item.is_media:
                    result_queue.put(
                        HashResult(
                            path=work_item.path,
                            owner=work_item.owner,
                            tags=work_item.tags,
                            sidecar_path=work_item.sidecar_path,
                            is_media=False,
                        )
                    )
                    continue

                similarity_hash = compute_hash(work_item.path)
                if not similarity_hash:
                    result_queue.put(
                        HashResult(
                            path=work_item.path,
                            owner=work_item.owner,
                            tags=work_item.tags,
                            sidecar_path=work_item.sidecar_path,
                            is_media=True,
                            error="hash_failed",
                        )
                    )
                    continue

                date_taken, date_source = resolve_date(work_item.path, work_item.sidecar_path)
                width, height = get_image_resolution(work_item.path)
                result_queue.put(
                    HashResult(
                        path=work_item.path,
                        owner=work_item.owner,
                        tags=work_item.tags,
                        sidecar_path=work_item.sidecar_path,
                        is_media=True,
                        similarity_hash=similarity_hash,
                        date_taken=date_taken,
                        date_source=date_source,
                        width=width,
                        height=height,
                    )
                )
            except Exception as e:
                result_queue.put(
                    HashResult(
                        path=work_item.path,
                        owner=work_item.owner,
                        tags=work_item.tags,
                        sidecar_path=work_item.sidecar_path,
                        is_media=work_item.is_media,
                        error=str(e),
                    )
                )
    except KeyboardInterrupt:
        pass


def copy_worker_process(
    copy_queue: mp.Queue,
    result_queue: mp.Queue,
    config_dict: dict,
    worker_id: int,
):
    """Worker process that copies files and writes EXIF (phase 2)."""
    output_root = Path(config_dict["output_root"])
    layout = StorageLayout(config_dict["storage_layout"])
    copy_non_media = config_dict.get("copy_non_media", True)
    dry_run = config_dict.get("dry_run", False)
    modify_exif = config_dict.get("modify_exif", True)
    copy_sidecar = config_dict.get("copy_sidecar", False)

    fmt = FilenameFormat(
        include_time=config_dict.get("fmt_include_time", True),
        year_format=YearFormat(config_dict.get("fmt_year_format", "YYYY")),
        month_format=MonthFormat(config_dict.get("fmt_month_format", "MM")),
        day_format=DayFormat(config_dict.get("fmt_day_format", "DD")),
        include_tags=config_dict.get("fmt_include_tags", True),
        max_tags=config_dict.get("fmt_max_tags", 5),
    )

    exif_batch = None
    if modify_exif and not dry_run:
        try:
            exif_batch = ExifToolBatch()
        except Exception:
            exif_batch = None

    try:
        while True:
            try:
                item = copy_queue.get(timeout=1)
            except Empty:
                continue

            if item == STOP_SENTINEL:
                break

            kind = item.get("kind")

            if kind == "non_media":
                if not copy_non_media:
                    continue
                source_path = item["source_path"]
                owner = item["owner"]
                try:
                    non_media_dir = output_root / owner / "non_media"
                    if not dry_run:
                        non_media_dir.mkdir(parents=True, exist_ok=True)
                    dest = non_media_dir / source_path.name
                    counter = 0
                    while not dry_run and dest.exists():
                        counter += 1
                        dest = non_media_dir / f"{source_path.stem}_{counter}{source_path.suffix}"
                    if not dry_run:
                        shutil.copy2(source_path, dest)
                    record = MediaRecord(
                        similarity_hash=f"non_media:{source_path}",
                        canonical_path=str(dest),
                        owner=owner,
                        date_taken=None,
                        date_source="none",
                        tags=[],
                        source_paths=[str(source_path)],
                        status="non_media",
                        notes=None,
                    )
                    result_queue.put(
                        CopyResult(
                            record=record,
                            pending_op_id=None,
                            replace_existing_path=None,
                            action="non_media",
                        )
                    )
                except Exception as e:
                    result_queue.put(
                        CopyResult(
                            record=MediaRecord(
                                similarity_hash=f"error:{source_path}",
                                canonical_path="",
                                owner=owner,
                                date_taken=None,
                                date_source="error",
                                tags=[],
                                source_paths=[str(source_path)],
                                status="error",
                                notes=str(e),
                            ),
                            pending_op_id=None,
                            replace_existing_path=None,
                            action="error",
                            error=str(e),
                        )
                    )
                continue

            if kind == "media":
                plan: CopyPlan = item["plan"]
                pending_op_id = item.get("pending_op_id")
                try:
                    canonical_path = plan.canonical_path
                    if not dry_run:
                        canonical_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(plan.source_path, canonical_path)

                    sidecar_extra = False
                    if plan.sidecar_path:
                        sidecar = load_sidecar(plan.sidecar_path)
                        if sidecar and modify_exif:
                            try:
                                if exif_batch:
                                    exif_batch.queue_write(canonical_path, sidecar, plan.filename_tags)
                                else:
                                    sidecar_to_exif(canonical_path, sidecar, plan.filename_tags)
                            except Exception:
                                pass
                        if sidecar:
                            sidecar_extra = sidecar_has_extras(sidecar)
                            if (copy_sidecar or sidecar_extra) and not dry_run:
                                target_sidecar = canonical_path.with_suffix(
                                    canonical_path.suffix + ".supplemental-metadata.json"
                                )
                                if not target_sidecar.exists():
                                    shutil.copy2(plan.sidecar_path, target_sidecar)
                    elif plan.filename_tags and modify_exif:
                        try:
                            if exif_batch:
                                exif_batch.queue_write(canonical_path, {}, plan.filename_tags)
                            else:
                                sidecar_to_exif(canonical_path, {}, plan.filename_tags)
                        except Exception:
                            pass

                    if exif_batch and exif_batch._pending >= 20:
                        exif_batch.flush_and_wait()

                    record = MediaRecord(
                        similarity_hash=plan.similarity_hash,
                        canonical_path=str(canonical_path),
                        owner=plan.owner,
                        date_taken=plan.date_taken.isoformat() if plan.date_taken else None,
                        date_source=plan.date_source,
                        tags=sorted(set(plan.record_tags)),
                        source_paths=sorted(set(plan.record_source_paths)),
                        status=plan.status,
                        notes=None if not sidecar_extra else "sidecar_copied",
                        width=plan.width,
                        height=plan.height,
                    )
                    result_queue.put(
                        CopyResult(
                            record=record,
                            pending_op_id=pending_op_id,
                            replace_existing_path=plan.replace_existing_path,
                            action="insert",
                        )
                    )
                except Exception as e:
                    result_queue.put(
                        CopyResult(
                            record=MediaRecord(
                                similarity_hash=f"error:{plan.source_path}",
                                canonical_path="",
                                owner=plan.owner,
                                date_taken=None,
                                date_source="error",
                                tags=plan.record_tags,
                                source_paths=[str(plan.source_path)],
                                status="error",
                                notes=str(e),
                            ),
                            pending_op_id=pending_op_id,
                            replace_existing_path=plan.replace_existing_path,
                            action="error",
                            error=str(e),
                        )
                    )
    except KeyboardInterrupt:
        pass
    finally:
        if exif_batch:
            try:
                exif_batch.flush_and_wait()
            except Exception:
                pass
            exif_batch.close()


def process_media_mp(
    config: AppConfig,
    logger: logging.Logger,
    limit: Optional[int] = None,
    recursive: bool = True,
    num_workers: int = 4,
) -> dict:
    """Process media files using multiprocessing (two-phase: hash then copy)."""

    temp_db_file = None
    database = None
    if config.dry_run:
        logger.info("DRY RUN MODE - using temporary database (will be deleted after run)")
        temp_db_file = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        db_path = Path(temp_db_file.name)
        temp_db_file.close()
        known_sources = set()
    else:
        db_path = config.resolve_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        database = MediaDatabase(db_path)

        # Recover from pending operations (crash recovery)
        pending = database.get_pending_operations()
        if pending:
            logger.warning("Found %d pending operations from previous run, cleaning up...", len(pending))
            for op in pending:
                if op.operation == "copy" and op.target_path:
                    try:
                        Path(op.target_path).unlink(missing_ok=True)
                    except Exception:
                        pass
            cleared = database.clear_all_pending_operations()
            logger.info("Cleared %d pending operations", cleared)

        logger.info("Loading known source paths from database...")
        raw_known_sources = database.get_all_source_paths()
        known_sources = {os.path.normpath(p) for p in raw_known_sources}
        logger.info("Found %d known source paths", len(known_sources))

    copy_non_media = config.copy_non_media
    duplicate_policy = config.duplicate_policy.value
    fmt = config.filename_format or FilenameFormat()

    logger.info("Scanning input directories...")
    work_items: list[WorkItem] = []
    skipped_known = 0
    media_count = 0
    non_media_count = 0

    for input_root in config.input_roots:
        if not input_root.path.exists():
            logger.warning("Input root missing: %s", input_root.path)
            continue
        debug_logged = 0
        for item in iter_work_items(input_root.path, input_root.owner, recursive=recursive, include_non_media=copy_non_media):
            normalized_path = os.path.normpath(str(item.path))
            if normalized_path in known_sources:
                skipped_known += 1
                continue
            if debug_logged < 3 and known_sources:
                sample_known = next(iter(known_sources))
                logger.debug("Path comparison - item: %r, sample known: %r", normalized_path, sample_known)
                debug_logged += 1
            work_items.append(item)
            if item.is_media:
                media_count += 1
            else:
                non_media_count += 1
            if limit and len(work_items) >= limit:
                break
        if limit and len(work_items) >= limit:
            break

    total_items = media_count
    logger.info(
        "Found %d media files to process (%d non-media files will also be handled, %d skipped_known)",
        media_count,
        non_media_count,
        skipped_known,
    )

    if total_items == 0:
        logger.info("============================================================")
        logger.info("PROCESSING SUMMARY")
        logger.info("============================================================")
        logger.info("No new files to process (all already in database)")
        logger.info("============================================================")
        return {"processed": 0, "skipped": 0, "errors": 0, "non_media": 0, "skipped_known": skipped_known}

    # Phase 1: hash and metadata extraction
    hash_queue = mp.Queue(maxsize=1000)
    hash_result_queue = mp.Queue()

    config_dict = {
        "output_root": str(config.output_root),
        "storage_layout": config.storage_layout.value,
        "copy_non_media": copy_non_media,
        "dry_run": config.dry_run,
        "modify_exif": config.modify_exif,
        "copy_sidecar": config.copy_sidecar,
        "fmt_include_time": fmt.include_time,
        "fmt_year_format": fmt.year_format.value,
        "fmt_month_format": fmt.month_format.value,
        "fmt_day_format": fmt.day_format.value,
        "fmt_include_tags": fmt.include_tags,
        "fmt_max_tags": fmt.max_tags,
    }

    hash_workers = []
    for i in range(num_workers):
        p = mp.Process(target=worker_process, args=(hash_queue, hash_result_queue, config_dict, i))
        p.start()
        hash_workers.append(p)

    logger.info("Phase 1: hashing with %d workers...", num_workers)
    
    # Feed work items in a non-blocking way to avoid deadlock
    # Put items gradually while collecting results
    work_index = 0
    items_sent = 0
    sentinels_sent = 0

    hash_results: list[HashResult] = []
    expected_results = len(work_items)
    collected = 0
    last_progress = 0
    
    # Process results while feeding work items to avoid queue deadlock
    while collected < expected_results or items_sent < len(work_items):
        # Try to send more work items (non-blocking)
        while work_index < len(work_items):
            try:
                hash_queue.put_nowait(work_items[work_index])
                work_index += 1
                items_sent += 1
            except Full:
                break  # Queue full, will try again later
        
        # Send sentinels if all work items are sent
        if items_sent >= len(work_items) and sentinels_sent < num_workers:
            try:
                hash_queue.put_nowait(STOP_SENTINEL)
                sentinels_sent += 1
            except Full:
                pass
        
        # Try to collect results (with timeout to allow feeding more work)
        try:
            result: HashResult = hash_result_queue.get(timeout=0.1)
            hash_results.append(result)
            collected += 1
            # Log progress every 100 files or 10%
            if collected - last_progress >= 100 or (collected * 10 // expected_results) > (last_progress * 10 // expected_results):
                pct = collected * 100 // expected_results
                logger.info("Phase 1 progress: %d/%d (%d%%) hashed", collected, expected_results, pct)
                last_progress = collected
        except Empty:
            continue

    for p in hash_workers:
        try:
            p.join(timeout=10)
        except (KeyboardInterrupt, BrokenPipeError):
            p.terminate()

    # Phase 1 analysis: dedupe and build copy plan
    processed = 0
    skipped = 0
    errors = 0
    non_media = 0

    non_media_plans: list[dict] = []
    media_candidates: dict[str, list[HashResult]] = {}

    for result in hash_results:
        if result.error:
            errors += 1
            continue
        if not result.is_media:
            non_media += 1
            if copy_non_media:
                non_media_plans.append({"kind": "non_media", "source_path": result.path, "owner": result.owner})
            continue
        if not result.similarity_hash:
            errors += 1
            continue
        media_candidates.setdefault(result.similarity_hash, []).append(result)

    expanded_candidates: dict[str, list[HashResult]] = {}
    for sim_hash, group in media_candidates.items():
        if sim_hash.startswith("phash:") and len(group) > 1:
            sha_groups: dict[str, list[HashResult]] = {}
            for item in group:
                sha = sha256_file(item.path)
                sha_groups.setdefault(sha, []).append(item)
            if len(sha_groups) > 1:
                for sha, items in sha_groups.items():
                    expanded_candidates.setdefault(f"{sim_hash}|sha256:{sha}", []).extend(items)
            else:
                expanded_candidates.setdefault(sim_hash, []).extend(group)
        else:
            expanded_candidates.setdefault(sim_hash, []).extend(group)

    copy_plans: list[CopyPlan] = []

    for sim_hash, group in expanded_candidates.items():
        existing = None
        final_hash = sim_hash
        if database and sim_hash.startswith("phash:"):
            existing = database.get_by_hash(sim_hash)
            if existing and existing.canonical_path:
                try:
                    if not verify_hash_collision(Path(existing.canonical_path), group[0].path):
                        sha = sha256_file(group[0].path)
                        final_hash = f"{sim_hash}|sha256:{sha}"
                        existing = database.get_by_hash(final_hash)
                except Exception:
                    pass
        elif database:
            existing = database.get_by_hash(sim_hash)

        # Choose candidate
        def pixels(item: HashResult) -> int:
            return (item.width * item.height) if item.width and item.height else 0

        chosen = group[0]
        if duplicate_policy == "keep-higher-resolution":
            chosen = max(group, key=pixels)

        merged_tags = sorted({t for item in group for t in item.tags})
        merged_sources = sorted({str(item.path) for item in group})

        if existing and database and duplicate_policy == "keep-higher-resolution":
            existing_pixels = (existing.width * existing.height) if existing.width and existing.height else 0
            chosen_pixels = pixels(chosen)
            if chosen_pixels > existing_pixels:
                merged_tags = sorted(set(merged_tags).union(existing.tags))
                merged_sources = sorted(set(merged_sources).union(existing.source_paths))
                output_dir = build_output_dir(config.output_root, chosen.owner, chosen.date_taken, config.storage_layout)
                canonical_path = find_unique_filename(output_dir, chosen.date_taken, chosen.tags, chosen.path.suffix, fmt)
                status = "ok" if chosen.date_taken else "missing_date"
                copy_plans.append(
                    CopyPlan(
                        source_path=chosen.path,
                        owner=chosen.owner,
                        filename_tags=chosen.tags,
                        record_tags=merged_tags,
                        record_source_paths=merged_sources,
                        sidecar_path=chosen.sidecar_path,
                        date_taken=chosen.date_taken,
                        date_source=chosen.date_source,
                        similarity_hash=final_hash,
                        width=chosen.width,
                        height=chosen.height,
                        status=status,
                        canonical_path=canonical_path,
                        replace_existing_path=Path(existing.canonical_path),
                    )
                )
                if len(group) > 1:
                    skipped += len(group) - 1
            else:
                database.update_existing(final_hash, merged_tags, merged_sources)
                skipped += len(group)
            continue

        if existing and database:
            database.update_existing(final_hash, merged_tags, merged_sources)
            skipped += len(group)
            continue

        output_dir = build_output_dir(config.output_root, chosen.owner, chosen.date_taken, config.storage_layout)
        canonical_path = find_unique_filename(output_dir, chosen.date_taken, chosen.tags, chosen.path.suffix, fmt)
        status = "ok" if chosen.date_taken else "missing_date"
        copy_plans.append(
            CopyPlan(
                source_path=chosen.path,
                owner=chosen.owner,
                filename_tags=chosen.tags,
                record_tags=merged_tags,
                record_source_paths=merged_sources,
                sidecar_path=chosen.sidecar_path,
                date_taken=chosen.date_taken,
                date_source=chosen.date_source,
                similarity_hash=final_hash,
                width=chosen.width,
                height=chosen.height,
                status=status,
                canonical_path=canonical_path,
            )
        )
        if len(group) > 1:
            skipped += len(group) - 1

    if config.dry_run:
        processed = len(copy_plans)
        result = {
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
            "non_media": non_media,
            "skipped_known": skipped_known,
        }
        return result

    # Phase 2: copy and EXIF writing
    copy_queue = mp.Queue(maxsize=1000)
    copy_result_queue = mp.Queue()

    copy_workers = []
    for i in range(num_workers):
        p = mp.Process(target=copy_worker_process, args=(copy_queue, copy_result_queue, config_dict, i))
        p.start()
        copy_workers.append(p)

    logger.info("Phase 2: copying with %d workers...", num_workers)

    for plan in copy_plans:
        pending_op_id = None
        if database:
            pending_op_id = database.add_pending_operation(
                str(plan.source_path),
                str(plan.canonical_path),
                plan.similarity_hash,
                "copy",
            )
        copy_queue.put({"kind": "media", "plan": plan, "pending_op_id": pending_op_id})

    for item in non_media_plans:
        copy_queue.put(item)

    for _ in range(num_workers):
        try:
            copy_queue.put(STOP_SENTINEL)
        except (KeyboardInterrupt, BrokenPipeError):
            pass

    expected_copy_results = len(copy_plans) + len(non_media_plans)
    collected = 0
    last_progress = 0
    while collected < expected_copy_results:
        try:
            result: CopyResult = copy_result_queue.get(timeout=1)
        except Empty:
            continue
        collected += 1
        # Log progress every 100 files or 10%
        if collected - last_progress >= 100 or (expected_copy_results > 0 and (collected * 10 // expected_copy_results) > (last_progress * 10 // expected_copy_results)):
            pct = collected * 100 // expected_copy_results if expected_copy_results > 0 else 100
            logger.info("Phase 2 progress: %d/%d (%d%%) copied", collected, expected_copy_results, pct)
            last_progress = collected
        if result.action == "insert":
            if database:
                database.upsert(result.record)
                if result.pending_op_id:
                    database.complete_pending_operation(result.pending_op_id)
            if result.replace_existing_path:
                try:
                    result.replace_existing_path.unlink(missing_ok=True)
                except Exception:
                    pass
            processed += 1
        elif result.action == "non_media":
            pass
        elif result.action == "error":
            errors += 1
            if database and result.pending_op_id:
                database.complete_pending_operation(result.pending_op_id)

    for p in copy_workers:
        try:
            p.join(timeout=10)
        except (KeyboardInterrupt, BrokenPipeError):
            p.terminate()

    result = {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "non_media": non_media,
        "skipped_known": skipped_known,
    }

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

    expected = media_count
    actual = result["processed"] + result["skipped"] + result["errors"]
    if expected == actual:
        logger.info("Validation: All files accounted for ✓")
    else:
        logger.warning("Validation MISMATCH: expected=%d, actual=%d", expected, actual)
    logger.info("============================================================")

    if config.dry_run and temp_db_file:
        try:
            db_path.unlink(missing_ok=True)
            logger.info("Temporary database cleaned up")
        except Exception as e:
            logger.warning("Failed to clean up temporary database: %s", e)

    if database:
        database.close()

    return result
