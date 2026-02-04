#!/usr/bin/env python3
"""
WhatsApp Image Sorter - Separates memes (images with text) from family photos
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Tuple
from PIL import Image
import pytesseract
from tqdm import tqdm


def detect_text_in_image(image_path: str, text_threshold: int = 20) -> Tuple[bool, int]:
    """
    Detect if an image contains significant text (likely a meme).
    
    Args:
        image_path: Path to the image file
        text_threshold: Minimum character count to consider as "has text"
        
    Returns:
        Tuple of (has_significant_text, character_count)
    """
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(img)
        
        # Remove whitespace and count meaningful characters
        text_clean = ''.join(text.split())
        char_count = len(text_clean)
        
        # Consider it a meme if it has significant text
        has_text = char_count >= text_threshold
        
        return has_text, char_count
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False, 0


def separate_whatsapp_images(source_dir: str, dry_run: bool = False):
    """
    Separate WhatsApp images into memes and photos subdirectories.
    
    Args:
        source_dir: Directory containing WhatsApp images
        dry_run: If True, only analyze without moving files
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Directory not found: {source_dir}")
        return
    
    # Create subdirectories
    photos_dir = source_path / "photos"
    memes_dir = source_path / "memes"
    
    if not dry_run:
        photos_dir.mkdir(exist_ok=True)
        memes_dir.mkdir(exist_ok=True)
        print(f"Created subdirectories:")
        print(f"  - {photos_dir}")
        print(f"  - {memes_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_files = [
        f for f in source_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    print(f"\nFound {len(image_files)} images to process")
    
    if not image_files:
        print("No images found!")
        return
    
    # Process images
    meme_count = 0
    photo_count = 0
    error_count = 0
    
    print("\nAnalyzing images for text content...")
    
    for img_file in tqdm(image_files, desc="Processing"):
        try:
            has_text, char_count = detect_text_in_image(str(img_file))
            
            if has_text:
                destination = memes_dir / img_file.name
                category = "meme"
                meme_count += 1
            else:
                destination = photos_dir / img_file.name
                category = "photo"
                photo_count += 1
            
            if dry_run:
                # Just print what would happen
                tqdm.write(f"[{category}] {img_file.name} ({char_count} chars)")
            else:
                # Move the file
                shutil.move(str(img_file), str(destination))
                
        except Exception as e:
            error_count += 1
            tqdm.write(f"Error processing {img_file.name}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Photos (no text):     {photo_count}")
    print(f"Memes (with text):    {meme_count}")
    print(f"Errors:               {error_count}")
    print(f"Total processed:      {len(image_files)}")
    
    if dry_run:
        print("\n[DRY RUN] No files were moved. Run without --dry-run to apply changes.")
    else:
        print(f"\nFiles organized into:")
        print(f"  - {photos_dir}")
        print(f"  - {memes_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Separate WhatsApp images into memes and photos based on text detection'
    )
    parser.add_argument(
        'directory',
        help='Path to WhatsApp Images directory'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze only, do not move files'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=20,
        help='Minimum character count to classify as meme (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Check if tesseract is installed
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        print("Error: tesseract-ocr is not installed.")
        print("Please install it:")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("  Fedora: sudo dnf install tesseract")
        print("  Arch: sudo pacman -S tesseract")
        sys.exit(1)
    
    separate_whatsapp_images(args.directory, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
