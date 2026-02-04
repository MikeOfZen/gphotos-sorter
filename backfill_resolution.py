#!/usr/bin/env python3
"""Backfill width and height for existing database records."""
import sqlite3
import sys
from pathlib import Path
from PIL import Image

def backfill_resolution(db_path: Path):
    """Scan existing files and populate their resolution data."""
    conn = sqlite3.connect(str(db_path))
    
    # Get count of records needing backfill
    cursor = conn.execute("SELECT COUNT(*) FROM media WHERE width IS NULL")
    total = cursor.fetchone()[0]
    print(f"Found {total} records without resolution data")
    
    if total == 0:
        print("✅ All records already have resolution data")
        conn.close()
        return
    
    # Get records to update
    cursor = conn.execute("SELECT id, canonical_path FROM media WHERE width IS NULL")
    records = cursor.fetchall()
    
    updated = 0
    failed = 0
    
    for row_id, path in records:
        try:
            path_obj = Path(path)
            # Only process image files, skip videos
            if path_obj.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}:
                continue
                
            with Image.open(path) as img:
                width, height = img.size
                conn.execute("UPDATE media SET width = ?, height = ? WHERE id = ?", 
                            (width, height, row_id))
                updated += 1
                if updated % 1000 == 0:
                    conn.commit()
                    print(f"Progress: {updated}/{total} ({100*updated/total:.1f}%)")
        except Exception as e:
            failed += 1
            if failed <= 10:  # Only show first 10 errors
                print(f"Failed to read {path}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Backfill complete:")
    print(f"   Updated: {updated}")
    print(f"   Failed: {failed}")
    print(f"   Total: {total}")

if __name__ == "__main__":
    db_path = Path('/mnt/offload/photos/media.sqlite')
    backfill_resolution(db_path)
