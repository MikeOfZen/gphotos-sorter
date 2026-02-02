# gphotos-sorter

A CLI tool for ingesting and organizing Google Photos takeout exports and other photo archives. It deduplicates media using perceptual hashing, preserves album metadata as EXIF tags, and organizes files into a clean folder structure.

## Features

- **Deduplication**: Uses perceptual image hashing to detect duplicate photos even if they've been resized or recompressed
- **Album Preservation**: Extracts album names from folder structure and writes them as EXIF keywords/tags
- **Google Photos Sidecar Support**: Reads JSON sidecar files from Google Photos takeout to extract:
  - Original capture date/time
  - GPS coordinates (written to EXIF)
  - Descriptions
  - Album information
- **Flexible Organization**: Organize output by year/month, year-month, or single folder
- **Configurable Filenames**: Customize filename format with year, month, day, time, and album tags
- **Multi-owner Support**: Process photos from multiple sources with different owner labels
- **SQLite Database**: Track all processed media with metadata for later querying
- **Multiprocessing**: Speed up processing with multiple worker processes
- **Dry-run Mode**: Preview what would be done without copying files

## Installation

### Prerequisites

- Python 3.10+
- exiftool (required for EXIF manipulation)

```bash
# Install exiftool
# On Debian/Ubuntu:
sudo apt-get install libimage-exiftool-perl

# On macOS:
brew install exiftool

# On Fedora:
sudo dnf install perl-Image-ExifTool
```

### Install gphotos-sorter

```bash
# Clone the repository
git clone https://github.com/yourusername/gphotos-sorter.git
cd gphotos-sorter

# Create virtual environment and install
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or with uv:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Quick Start

```bash
# Basic usage - process Google Photos takeout
gphotos-sorter run \
  --input "/path/to/Google Photos" \
  --output "/path/to/organized/photos" \
  --owner "MyName"

# Dry run first to see what would be done
gphotos-sorter run \
  --input "/path/to/Google Photos" \
  --output "/path/to/organized/photos" \
  --dry-run

# Use multiprocessing for faster processing
gphotos-sorter run \
  --input "/path/to/Google Photos" \
  --output "/path/to/organized/photos" \
  --workers 8

# Limit to first 100 files for testing
gphotos-sorter run \
  --input "/path/to/Google Photos" \
  --output "/path/to/organized/photos" \
  --limit 100
```

## Command Line Options

### Required Options

| Option | Description |
|--------|-------------|
| `-i, --input PATH` | Input path(s) to process. Can specify multiple times. |
| `-O, --output PATH` | Output root folder for organized media (required) |

### Organization Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --owner TEXT` | "Mine" | Owner label for input paths |
| `-l, --layout` | "year-month" | Storage layout: `single`, `year/month`, or `year-month` |
| `--db PATH` | `<output>/media.sqlite` | SQLite database path |

### Processing Options

| Option | Default | Description |
|--------|---------|-------------|
| `-n, --limit INT` | None | Limit number of files to process |
| `--no-recursive` | False | Only process files directly in input folders |
| `-w, --workers INT` | 1 | Number of worker processes (>1 enables multiprocessing) |
| `-d, --dry-run` | False | Don't copy files, just show what would be done |

### Filename Format Options

| Option | Default | Description |
|--------|---------|-------------|
| `--year-format` | "YYYY" | Year format: `YYYY` (2021) or `YY` (21) |
| `--month-format` | "MM" | Month format: `MM` (06), `name` (June), or `short` (Jun) |
| `--day-format` | "DD" | Day format: `DD` (15) or `weekday` (15_Tuesday) |
| `--no-time` | False | Exclude time (HHMMSS) from filename |
| `--no-tags` | False | Exclude album tags from filename |
| `--max-tags INT` | None | Maximum number of tags (default: no limit) |

### File Handling Options

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-non-media` | False | Skip non-media files (default: copy with warning) |
| `--copy-sidecar` | False | Copy sidecar JSON files alongside media |
| `--no-exif` | False | Skip writing metadata to EXIF tags |

### Logging Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable debug logging |
| `-vv` | Enable trace-level logging with function names |

## Output Structure

### Default (year-month layout)

```
output/
├── MyName/
│   ├── 2021-06/
│   │   ├── 20210615_143022_Vacation.jpg
│   │   └── 20210615_143045_Vacation.jpg
│   ├── 2021-12/
│   │   └── 20211225_120000_Christmas.jpg
│   ├── unknown/              # Files without dates
│   │   └── FamilyPhoto.jpg
│   └── non_media/            # Non-media files
│       └── metadata.json
└── media.sqlite              # Tracking database
```

### Filename Format Examples

```bash
# Default: YYYYMMDD_HHMMSS_Tags.ext
20210615_143022_Vacation_Beach.jpg

# With --month-format name:
2021_June_15_143022_Vacation_Beach.jpg

# With --year-format YY --month-format short:
21_Jun_15_143022_Vacation_Beach.jpg

# With --day-format weekday:
20210615_Tuesday_143022_Vacation_Beach.jpg

# With --no-time:
20210615_Vacation_Beach.jpg

# With --no-tags:
20210615_143022.jpg
```

## How It Works

### 1. Discovery Phase

The tool scans input directories and for each file:
- Identifies if it's a media file (image/video) or non-media
- Extracts folder names as potential album tags
- Looks for Google Photos sidecar JSON files

### 2. Deduplication

For each media file:
- Computes a perceptual hash (pHash) that's resistant to minor changes
- Checks if this hash already exists in the database
- If duplicate: updates the record with new source path and tags
- If new: continues to processing

### 3. Date Resolution

Attempts to determine when the photo was taken, in order:
1. EXIF DateTimeOriginal from the image
2. Google Photos sidecar JSON (`photoTakenTime`)
3. Folder name parsing (e.g., "2021-06-15" or "Photos from 2021")

### 4. Metadata Preservation

Writes metadata to the copied file's EXIF:
- Album names → EXIF Keywords/Subject tags
- GPS coordinates → EXIF GPS tags
- Description → EXIF Description

### 5. Organization

Copies the file to the output directory:
- Creates folder structure based on date and layout setting
- Generates filename from date, time, and album tags
- Handles conflicts by appending counter

## Database

The SQLite database tracks all processed media with:

| Field | Description |
|-------|-------------|
| `similarity_hash` | Perceptual hash for deduplication |
| `canonical_path` | Path to the organized copy |
| `owner` | Owner label |
| `date_taken` | When the photo was taken |
| `date_source` | Where the date came from (exif/sidecar/folder) |
| `tags` | Album/folder tags |
| `source_paths` | All original paths (for duplicates) |
| `status` | ok, missing_date, error |

## Examples

### Process Multiple Input Directories

```bash
gphotos-sorter run \
  --input "/backup/google-photos" \
  --input "/backup/iphone-photos" \
  --output "/organized/photos" \
  --owner "Family"
```

### Process Different Owners

```bash
# First person
gphotos-sorter run \
  --input "/backup/alice-photos" \
  --output "/family-archive" \
  --owner "Alice"

# Second person
gphotos-sorter run \
  --input "/backup/bob-photos" \
  --output "/family-archive" \
  --owner "Bob"
```

### Minimal Filenames

```bash
gphotos-sorter run \
  --input "/photos" \
  --output "/archive" \
  --no-tags \
  --no-time
# Results in: 20210615.jpg
```

### Verbose Processing

```bash
gphotos-sorter run \
  --input "/photos" \
  --output "/archive" \
  -vv
# Shows detailed debug output with function names
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Uses [imagehash](https://github.com/JohannesBuchner/imagehash) for perceptual hashing
- Uses [exiftool](https://exiftool.org/) for EXIF manipulation
- Built with [Typer](https://typer.tiangolo.com/) for CLI
