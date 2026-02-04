# Database Schema Migration - Completed

## Changes Made

### 1. Schema Updates ✅
Added three new columns to the `media` table:
- `faces_hashes` (TEXT): JSON array for face recognition hashes (future feature)
- `ai_desc` (TEXT): AI-generated image descriptions (future feature)  
- `objects` (TEXT): JSON array of detected objects (future feature)

### 2. Automatic Migration ✅
- Added `_migrate_schema()` method that runs on database initialization
- Checks for missing columns and adds them dynamically
- Backward compatible: works with both new and existing databases
- No data loss: existing records preserved

### 3. Updated Code ✅
Modified files:
- `uncloud/persistence/database.py`:
  - Updated `MediaRecord` dataclass with new fields
  - Updated `upsert()` to handle new columns
  - Updated `_row_to_record()` to read new columns
  - Added `_migrate_schema()` for automatic migration

## Testing

```bash
# Test on existing database
✓ Migration applied to /mnt/offload/photos/.uncloud.db
✓ All columns present: id, canonical_path, similarity_hash, owner, 
  date_taken, tags, width, height, source_paths, created_at, updated_at,
  faces_hashes, ai_desc, objects

# Test on new database
✓ Schema created correctly with all columns
```

## Database Sync Strategy

See `SYNC_DESIGN.md` for the full design document covering:

### Filesystem as Source of Truth
- DB becomes an **index** of what exists on disk
- Can be rebuilt at any time from filesystem
- Handles manual filesystem changes gracefully

### Deduplication Strategy
- **Hash-based**: Uses similarity_hash as primary key
- **Source tracking**: Keeps array of all original source paths
- **Multi-source**: Multiple imports of same file tracked in `source_paths`

### Sync Operations
On startup, the tool should:
1. Remove DB entries for files no longer on disk
2. Index files on disk not yet in DB
3. Validate hash consistency

### Future Features
The schema is now ready for:
- Face recognition (faces_hashes)
- AI image captioning (ai_desc)
- Object detection (objects)

## Next Steps

### Phase 1: Fix Duplicate Creation Bug ⚠️
**Problem**: Files being copied with `_1` suffix even when hash exists in DB
- Root cause: Likely race condition in multiprocessing
- 1,934 duplicate files found (19.5 GB) before cleanup
- Need to investigate why `get_by_hash()` isn't preventing copies

**Proposed Solutions**:
1. Add file-level locking during hash check + copy
2. Use database transaction isolation
3. Add logging to trace duplicate creation
4. Consider using canonical_path as unique constraint instead of hash

### Phase 2: Implement Sync Validation
Add `--rebuild-index` CLI flag and sync validation:
```python
def validate_sync(output_dir: Path, repo: SQLiteMediaRepository):
    # Remove orphaned DB entries
    for record in repo.get_all_records():
        if not Path(record.canonical_path).exists():
            repo.delete(record.id)
    
    # Index untracked files
    for file in scan_media_files(output_dir):
        if not repo.has_path(file):
            hash_result = compute_hash(file)
            repo.index_file(file, hash_result)
```

### Phase 3: AI Features (Future)
- Integrate face detection library
- Add image captioning model
- Implement object detection
- Enable semantic search via embeddings

## Migration Guide

### For Users
No action needed! Migration happens automatically when you run:
```bash
uv run python -m uncloud [args]
```

The tool will:
1. Detect existing database
2. Check for missing columns
3. Add them automatically
4. Continue normal operation

### For Developers
To manually check migration status:
```python
from uncloud.persistence.database import SQLiteMediaRepository
from pathlib import Path

db = SQLiteMediaRepository(Path("your_db.sqlite"))
cursor = db._conn.cursor()
cursor.execute("PRAGMA table_info(media)")
print(cursor.fetchall())
```

## Rollback

If issues occur, you can rollback by:
1. Deleting the database: `rm /output/.uncloud.db`
2. Running the tool again to rebuild from scratch

The new columns are all nullable, so existing code continues to work even if migration fails.
