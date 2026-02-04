#!/usr/bin/env python3
"""One-time migration script to add width and height columns to existing databases."""
import sqlite3
import sys
from pathlib import Path


def migrate_database(db_path: Path) -> None:
    """Add width and height columns if they don't exist."""
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)
    
    conn = sqlite3.connect(str(db_path))
    
    # Check existing columns
    cursor = conn.execute("PRAGMA table_info(media)")
    columns = {row[1] for row in cursor.fetchall()}
    
    changes_made = False
    
    if "width" not in columns:
        print("Adding 'width' column...")
        conn.execute("ALTER TABLE media ADD COLUMN width INTEGER")
        changes_made = True
    else:
        print("'width' column already exists")
    
    if "height" not in columns:
        print("Adding 'height' column...")
        conn.execute("ALTER TABLE media ADD COLUMN height INTEGER")
        changes_made = True
    else:
        print("'height' column already exists")
    
    if changes_made:
        conn.commit()
        print("✅ Migration complete!")
    else:
        print("✅ Database already up to date")
    
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python migrate_db.py <path_to_database>")
        print("Example: python migrate_db.py /mnt/offload/photos/media.sqlite")
        sys.exit(1)
    
    db_path = Path(sys.argv[1])
    migrate_database(db_path)
