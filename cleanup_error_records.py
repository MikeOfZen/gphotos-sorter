#!/usr/bin/env python3
"""Remove error records from the database.

Error records have similarity_hash starting with 'error:' and were incorrectly
added to the database when files couldn't be processed (e.g., HDD disconnected).

This script removes them so the files can be reprocessed on the next run.
"""
import sqlite3
import sys
from pathlib import Path


def cleanup_error_records(db_path: Path, dry_run: bool = False) -> int:
    """Remove all error records from the database.
    
    Returns the number of records removed.
    """
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)
    
    conn = sqlite3.connect(str(db_path))
    
    # Count error records
    cursor = conn.execute(
        "SELECT COUNT(*) FROM media WHERE similarity_hash LIKE 'error:%'"
    )
    error_count = cursor.fetchone()[0]
    
    # Count source paths that will be freed
    cursor = conn.execute(
        "SELECT source_paths FROM media WHERE similarity_hash LIKE 'error:%'"
    )
    source_count = 0
    for row in cursor:
        if row[0]:
            import json
            paths = json.loads(row[0])
            source_count += len(paths)
    
    print(f"Found {error_count} error records with {source_count} source paths")
    
    if error_count == 0:
        print("✅ No error records to remove")
        conn.close()
        return 0
    
    if dry_run:
        print(f"DRY RUN: Would remove {error_count} error records")
        
        # Show sample of what would be removed
        cursor = conn.execute(
            "SELECT similarity_hash, source_paths FROM media WHERE similarity_hash LIKE 'error:%' LIMIT 5"
        )
        print("\nSample records that would be removed:")
        for hash, sources in cursor:
            print(f"  {hash[:50]}...")
            if sources:
                import json
                for path in json.loads(sources)[:2]:
                    print(f"    - {path}")
        
        conn.close()
        return 0
    
    # Remove error records
    print(f"Removing {error_count} error records...")
    conn.execute("DELETE FROM media WHERE similarity_hash LIKE 'error:%'")
    conn.commit()
    
    # Vacuum to reclaim space
    print("Vacuuming database...")
    conn.execute("VACUUM")
    
    conn.close()
    
    print(f"✅ Removed {error_count} error records")
    print(f"   {source_count} source paths are now available for reprocessing")
    
    return error_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove error records from gphotos-sorter database")
    parser.add_argument("db_path", type=Path, help="Path to media.sqlite database")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be removed without deleting")
    
    args = parser.parse_args()
    
    cleanup_error_records(args.db_path, dry_run=args.dry_run)
