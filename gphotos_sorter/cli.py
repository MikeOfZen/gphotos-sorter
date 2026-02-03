from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import typer

from .config import (
    AppConfig, InputRoot, StorageLayout, 
    FilenameFormat, YearFormat, MonthFormat, DayFormat
)
from .scanner import process_media
from .scanner_mp import process_media_mp

app = typer.Typer(
    add_completion=True,
    help="""Media ingestion and organization CLI.

Ingest Google Photos takeout and other photo archives, deduplicate media,
preserve album tags in EXIF, and store metadata in SQLite.
"""
)


def setup_logger(verbosity: int) -> logging.Logger:
    """Setup logger with verbosity level (0=INFO, 1=DEBUG, 2=TRACE)."""
    logger = logging.getLogger("gphotos_sorter")
    if verbosity >= 2:
        logger.setLevel(logging.DEBUG)
        # Add extra detailed formatting for -vv
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(funcName)s:%(lineno)d] %(message)s"))
    elif verbosity >= 1:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    else:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers = [handler]
    return logger



def check_exiftool() -> None:
    """Check that exiftool is installed and available."""
    if not shutil.which("exiftool"):
        typer.secho("ERROR: exiftool is required but not found in PATH.", fg=typer.colors.RED, err=True)
        typer.secho("Install it with: sudo apt-get install libimage-exiftool-perl", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)




@app.callback(invoke_without_command=True)
def main_command(
    ctx: typer.Context,
    input_path: Optional[List[Path]] = typer.Option(
        None, "--input", "-i",
        help="Input path(s) to process. Can specify multiple times."
    ),
    owner: str = typer.Option(
        "Mine", "--owner", "-o",
        help="Owner label for input paths"
    ),
    output_root: Optional[Path] = typer.Option(
        None, "--output", "-O",
        help="Output root folder (required)"
    ),
    storage_layout: StorageLayout = typer.Option(
        StorageLayout.year_dash_month, "--layout", "-l",
        help="Storage layout: single, year/month, or year-month"
    ),
    db_path: Optional[Path] = typer.Option(
        None, "--db",
        help="SQLite database path (default: output_root/media.sqlite)"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n",
        help="Limit number of files to process (for test runs)"
    ),
    no_recursive: bool = typer.Option(
        False, "--no-recursive",
        help="Only process files directly in input folders, not subdirectories"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w",
        help="Number of worker processes (1=single-threaded, >1=multiprocessing)"
    ),
    # Filename format options
    year_format: YearFormat = typer.Option(
        YearFormat.YYYY, "--year-format",
        help="Year format in filename: YYYY (2021) or YY (21)"
    ),
    month_format: MonthFormat = typer.Option(
        MonthFormat.MM, "--month-format",
        help="Month format in filename: MM (06), name (June), or short (Jun)"
    ),
    day_format: DayFormat = typer.Option(
        DayFormat.DD, "--day-format",
        help="Day format in filename: DD (15) or weekday (15_Tuesday)"
    ),
    no_time: bool = typer.Option(
        False, "--no-time",
        help="Exclude time (HHMMSS) from filename"
    ),
    no_tags: bool = typer.Option(
        False, "--no-tags",
        help="Exclude album tags from filename"
    ),
    max_tags: Optional[int] = typer.Option(
        None, "--max-tags",
        help="Maximum number of tags to include in filename (default: no limit)"
    ),
    # File handling options
    include_non_media: bool = typer.Option(
        False, "--include-non-media",
        help="Include non-media files (default: skip them)"
    ),
    copy_sidecar: bool = typer.Option(
        False, "--copy-sidecar",
        help="Copy sidecar JSON files alongside media"
    ),
    no_exif: bool = typer.Option(
        False, "--no-exif",
        help="Skip writing metadata to EXIF tags"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-d",
        help="Don't actually copy files, just show what would be done"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True,
        help="Increase verbosity (-v for debug, -vv for trace)"
    ),
) -> None:
    """Process and ingest media files.
    
    Scans input directories, deduplicates media using perceptual hashing,
    extracts dates from EXIF/sidecar/folder names, preserves album info as tags,
    copies unique files to organized output folders, and writes sidecar metadata to EXIF.
    """
    # If no arguments provided, show help
    if not input_path and not output_root:
        typer.echo(ctx.get_help())
        raise typer.Exit()
    
    check_exiftool()
    
    # Build config from CLI arguments
    if not input_path:
        typer.secho("ERROR: At least one --input path is required", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    if not output_root:
        typer.secho("ERROR: --output path is required", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    # Build filename format
    filename_format = FilenameFormat(
        include_time=not no_time,
        year_format=year_format,
        month_format=month_format,
        day_format=day_format,
        include_tags=not no_tags,
        max_tags=max_tags,
    )
    
    config = AppConfig(
        input_roots=[InputRoot(owner=owner, path=p.expanduser().resolve()) for p in input_path],
        output_root=output_root.expanduser().resolve(),
        storage_layout=storage_layout,
        db_path=db_path.expanduser().resolve() if db_path else None,
        filename_format=filename_format,
        copy_non_media=include_non_media,
        copy_sidecar=copy_sidecar,
        modify_exif=not no_exif,
        dry_run=dry_run,
    )
    
    logger = setup_logger(verbose)
    
    typer.echo(f"Output: {config.output_root}")
    typer.echo(f"Layout: {config.storage_layout.value}")
    typer.echo(f"DB: {config.resolve_db_path()}")
    for ir in config.input_roots:
        typer.echo(f"Input [{ir.owner}]: {ir.path}")
    if limit:
        typer.echo(f"Limit: {limit} files")
    if no_recursive:
        typer.echo("Mode: non-recursive (single folder only)")
    if workers > 1:
        typer.echo(f"Workers: {workers} (multiprocessing)")
    if dry_run:
        typer.secho("DRY RUN - no files will be copied", fg=typer.colors.YELLOW)
    
    # Show filename format if non-default
    fmt_parts = []
    if year_format != YearFormat.YYYY:
        fmt_parts.append(f"year={year_format.value}")
    if month_format != MonthFormat.MM:
        fmt_parts.append(f"month={month_format.value}")
    if day_format != DayFormat.DD:
        fmt_parts.append(f"day={day_format.value}")
    if no_time:
        fmt_parts.append("no-time")
    if no_tags:
        fmt_parts.append("no-tags")
    if max_tags is not None:
        fmt_parts.append(f"max-tags={max_tags}")
    if fmt_parts:
        typer.echo(f"Filename format: {', '.join(fmt_parts)}")
    
    # Show file handling options
    options = []
    if not include_non_media:
        options.append("skip-non-media")
    if copy_sidecar:
        options.append("copy-sidecar")
    if no_exif:
        options.append("no-exif")
    if options:
        typer.echo(f"Options: {', '.join(options)}")
    elif include_non_media:
        typer.echo("Non-media files: copy with warning")
    
    typer.echo("---")
    
    if workers > 1:
        process_media_mp(config, logger, limit=limit, recursive=not no_recursive, num_workers=workers)
    else:
        stats = process_media(config, logger, limit=limit, recursive=not no_recursive)
        
        # Show summary with color coding
        if stats["errors"] > 0:
            typer.secho(f"\n⚠ Completed with {stats['errors']} errors", fg=typer.colors.YELLOW)
        else:
            typer.secho(f"\n✓ Completed successfully", fg=typer.colors.GREEN)



def main() -> None:
    app()


if __name__ == "__main__":
    main()
