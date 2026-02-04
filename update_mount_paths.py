#!/usr/bin/env python3
"""Update mount paths in the database."""
import json
from pathlib import Path
from gphotos_sorter.db import MediaDatabase

db_path = Path('/mnt/offload/photos/media.sqlite')
db = MediaDatabase(db_path)

# Get all records
cursor = db.connection.execute('SELECT id, source_paths FROM media')
rows = cursor.fetchall()

updated = 0
for row_id, source_paths_json in rows:
    if source_paths_json:
        paths = json.loads(source_paths_json)
        # Replace veracrypt1 with veracrypt2
        new_paths = [p.replace('/media/veracrypt1/', '/media/veracrypt2/') for p in paths]
        if new_paths != paths:
            db.connection.execute(
                'UPDATE media SET source_paths = ? WHERE id = ?',
                (json.dumps(new_paths), row_id)
            )
            updated += 1

db.connection.commit()
db.close()

print(f'Updated {updated} records (veracrypt1 -> veracrypt2)')
