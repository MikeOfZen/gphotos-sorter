#!/usr/bin/env python3
"""Find and delete duplicate files by re-scanning filesystem and comparing hashes.

Since the rebuild only indexed unique files, we need to:
1. Load all known hashes from DB
2. Re-scan all files and compute their hashes  
3. Find files that hash to the same value as DB entries but are different paths
4. Delete the duplicate (keep the one in DB as the "winner")

Usage:
    python cleanup_duplicates.py /path/to/photos [--dry-run] [--verbose]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


# Import hash engine from uncloud
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from uncloud.hashing.cpu_hasher import CPUHashEngine


MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".heif",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp",
}


@dataclass
class FileInfo:
    """Information about a file."""
    path: Path
    width: int
    height: int
    file_size: int
    
    @property
    def resolution(self) -> int:
        """Total pixel count."""
        return self.width * self.height
    
    @property
    def exists(self) -> bool:
        """Check if file exists on disk."""
        return self.path.exists()


def get_duplicates(db_path: Path) -> dict[str, list[FileInfo]]:
    """Get all duplicate groups from the database.
    
    Returns a dict mapping similarity_hash -> list of FileInfo.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Find all hashes with more than one file
    query = """
        SELECT 
            similarity_hash,
            canonical_path,
            width,
            height
        FROM media
        WHERE similarity_hash IN (
            SELECT similarity_hash 
            FROM media 
            GROUP BY similarity_hash 
            HAVING COUNT(*) > 1
        )
        ORDER BY similarity_hash
    """
    
    duplicates: dict[str, list[FileInfo]] = defaultdict(list)
    
    for row in conn.execute(query):
        path = Path(row["canonical_path"])
        # Get file size from filesystem since DB may not have it
        try:
            file_size = path.stat().st_size if path.exists() else 0
        except OSError:
            file_size = 0
            
        info = FileInfo(
            path=path,
            width=row["width"] or 0,
            height=row["height"] or 0,
            file_size=file_size,
        )
        duplicates[row["similarity_hash"]].append(info)
    
    conn.close()
    return dict(duplicates)


def pick_winner(files: list[FileInfo]) -> tuple[FileInfo, list[FileInfo]]:
    """Pick the best file to keep (highest resolution, then largest file size).
    
    Returns (winner, losers).
    """
    # Sort by resolution descending, then file size descending
    sorted_files = sorted(
        files,
        key=lambda f: (f.resolution, f.file_size),
        reverse=True,
    )
    return sorted_files[0], sorted_files[1:]


def cleanup_duplicates(
    output_dir: Path,
    dry_run: bool = True,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """Find and delete duplicate files.
    
    Returns (groups_processed, files_deleted, bytes_freed).
    """
    db_path = output_dir / ".uncloud.db"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 0, 0, 0
    
    print(f"📂 Loading duplicates from: {db_path}")
    duplicates = get_duplicates(db_path)
    
    print(f"🔍 Found {len(duplicates)} duplicate groups")
    
    if not duplicates:
        print("✅ No duplicates to clean up!")
        return 0, 0, 0
    
    groups_processed = 0
    files_deleted = 0
    bytes_freed = 0
    missing_files = 0
    
    for hash_val, files in duplicates.items():
        winner, losers = pick_winner(files)
        
        for loser in losers:
            # Path is already absolute from DB
            full_path = loser.path
            
            if not full_path.exists():
                if verbose:
                    print(f"  ⚠ Already deleted: {loser.path.name}")
                missing_files += 1
                continue
            
            if verbose or dry_run:
                print(f"  🗑 {loser.path.name}")
                print(f"    ↳ Keep: {winner.path.name} ({winner.width}x{winner.height}, {format_bytes(winner.file_size)})")
                print(f"    ↳ Delete: {loser.path.name} ({loser.width}x{loser.height}, {format_bytes(loser.file_size)})")
            
            if not dry_run:
                try:
                    file_size = full_path.stat().st_size
                    full_path.unlink()
                    files_deleted += 1
                    bytes_freed += file_size
                except OSError as e:
                    print(f"  ❌ Error deleting {full_path}: {e}")
            else:
                # In dry-run, count what would be freed
                files_deleted += 1
                bytes_freed += loser.file_size
        
        groups_processed += 1
        
        # Progress indicator
        if groups_processed % 1000 == 0:
            print(f"  ⏳ Processed {groups_processed}/{len(duplicates)} groups...")
    
    return groups_processed, files_deleted, bytes_freed


def update_database(output_dir: Path, dry_run: bool = True) -> int:
    """Remove deleted files from the database.
    
    Returns number of records removed.
    """
    db_path = output_dir / ".uncloud.db"
    
    if dry_run:
        return 0
    
    conn = sqlite3.connect(db_path)
    
    # Find and remove records for files that no longer exist
    cursor = conn.execute("SELECT canonical_path FROM media")
    paths_to_delete = []
    
    for row in cursor:
        full_path = Path(row[0])
        if not full_path.exists():
            paths_to_delete.append(row[0])
    
    if paths_to_delete:
        placeholders = ",".join("?" * len(paths_to_delete))
        conn.execute(
            f"DELETE FROM media WHERE canonical_path IN ({placeholders})",
            paths_to_delete,
        )
        conn.commit()
    
    conn.close()
    return len(paths_to_delete)


def format_bytes(size: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean up duplicate files, keeping highest resolution versions."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to the photo output directory containing .uncloud.db",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be deleted without actually deleting (default)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete the duplicate files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show details for each duplicate",
    )
    
    args = parser.parse_args()
    
    # --execute overrides --dry-run
    dry_run = not args.execute
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
        print("   Use --execute to actually delete files\n")
    else:
        print("⚠️  EXECUTE MODE - Files will be permanently deleted!\n")
    
    groups, deleted, freed = cleanup_duplicates(
        output_dir=args.output_dir,
        dry_run=dry_run,
        verbose=args.verbose,
    )
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print(f"   Duplicate groups: {groups:,}")
    print(f"   Files {'to delete' if dry_run else 'deleted'}: {deleted:,}")
    print(f"   Space {'to free' if dry_run else 'freed'}: {format_bytes(freed)}")
    
    if not dry_run and deleted > 0:
        print("\n🗄 Updating database...")
        removed = update_database(args.output_dir, dry_run=False)
        print(f"   Removed {removed:,} database records")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
