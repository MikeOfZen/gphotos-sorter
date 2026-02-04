#!/usr/bin/env python3
"""
WhatsApp Image Sorter - Separates memes from family photos using CLIP ML model
Uses zero-shot classification with GPU acceleration for fast, accurate results.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Tuple, List
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


class ImageClassifier:
    """CLIP-based image classifier for memes vs photos"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        """
        Initialize the classifier with CLIP model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Define classification prompts
        self.categories = {
            "photo": [
                "a family photo",
                "a personal photograph",
                "a selfie",
                "a picture of people",
                "a landscape photo",
                "a photo without text"
            ],
            "meme": [
                "a meme with text overlay",
                "an internet meme",
                "a screenshot with text",
                "a meme image",
                "a social media post screenshot",
                "an image with caption text"
            ]
        }
        
        print(f"Model loaded successfully!")
    
    def classify_image(self, image_path: str) -> Tuple[str, float, dict]:
    """
    Classify an image using CLIP zero-shot classification.
    
    Args:
        image_path: Path to the image file
        model: CLIP model
        processor: CLIP processor
        device: torch device (cuda/cpu)
        text_prompts: List of text descriptions for classification
        
    Returns:
        tuple: (predicted_class_index, confidence, all_probabilities)
    """
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Prepare inputs
        inputs = processor(
            text=text_prompts,
            images=img,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get predicted class
        predicted_idx = probs.argmax().item()
        confidence = probs[0][predicted_idx].item()
        all_probs = probs[0].cpu().numpy().tolist()
        
        return predicted_idx, confidence, all_probs
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return -1, 0.0, []


def separate_whatsapp_images(
    source_dir: str,
    model_name: str = "openai/clip-vit-base-patch32",
    confidence_threshold: float = 0.55,
    dry_run: bool = False,
    verbose: bool = False
):
    """
    Separate WhatsApp images into photos and memes using CLIP.
    
    Args:
        source_dir: Directory containing WhatsApp images
        model_name: HuggingFace CLIP model to use
        confidence_threshold: Minimum confidence to classify (otherwise 'uncertain')
        dry_run: If True, only analyze without moving files
        verbose: Print details for each image
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Directory not found: {source_dir}")
        return
    
    # Setup CLIP model
    model, processor, device = setup_clip_model(model_name)
    
    # Define classification prompts
    # These can be customized based on your specific needs
    text_prompts = [
        "a family photo, a personal photograph, a selfie, people in real life, vacation photo",
        "a meme with text overlay, an internet meme, a screenshot with text, a funny image with captions"
    ]
    
    print("Classification categories:")
    print(f"  [0] PHOTOS: {text_prompts[0]}")
    print(f"  [1] MEMES:  {text_prompts[1]}")
    print(f"\nConfidence threshold: {confidence_threshold:.0%}")
    print(f"(Images below threshold will be placed in 'uncertain' folder)\n")
    
    # Create subdirectories
    photos_dir = source_path / "photos"
    memes_dir = source_path / "memes"
    uncertain_dir = source_path / "uncertain"
    
    if not dry_run:
        photos_dir.mkdir(exist_ok=True)
        memes_dir.mkdir(exist_ok=True)
        uncertain_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_files = [
        f for f in source_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_files)} images to process\n")
    
    if not image_files:
        print("No images found!")
        return
    
    # Statistics
    meme_count = 0
    photo_count = 0
    uncertain_count = 0
    error_count = 0
    
    # Process each image
    print("Analyzing images with AI...")
    for image_file in tqdm(image_files, desc="Classifying"):
        predicted_class, confidence, probs = classify_image_clip(
            str(image_file), model, processor, device, text_prompts
        )
        
        if predicted_class == -1:
            error_count += 1
            continue
        
        # Determine destination based on prediction and confidence
        if confidence < confidence_threshold:
            dest_dir = uncertain_dir
            category = "uncertain"
            uncertain_count += 1
        elif predicted_class == 0:
            dest_dir = photos_dir
            category = "photo"
            photo_count += 1
        else:
            dest_dir = memes_dir
            category = "meme"
            meme_count += 1
        
        if verbose or dry_run:
            tqdm.write(
                f"[{category.upper():9s}] {image_file.name:40s} "
                f"(confidence: {confidence:.1%}, photo: {probs[0]:.1%}, meme: {probs[1]:.1%})"
            )
        
        if not dry_run:
            dest_path = dest_dir / image_file.name
            shutil.move(str(image_file), str(dest_path))
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Photos (family/personal):    {photo_count:4d}  ({photo_count/len(image_files)*100:.1f}%)")
    print(f"Memes (text/screenshots):    {meme_count:4d}  ({meme_count/len(image_files)*100:.1f}%)")
    print(f"Uncertain (low confidence):  {uncertain_count:4d}  ({uncertain_count/len(image_files)*100:.1f}%)")
    if error_count > 0:
        print(f"Errors:                      {error_count:4d}")
    print(f"{'─'*70}")
    print(f"Total processed:             {len(image_files):4d}")
    
    if dry_run:
        print("\n[DRY RUN] No files were moved.")
        print("Run without --dry-run to apply changes.")
    else:
        print(f"\nFiles organized into:")
        print(f"  ✓ {photos_dir}")
        print(f"  ✓ {memes_dir}")
        if uncertain_count > 0:
            print(f"  ⚠ {uncertain_dir} (review these manually)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Separate WhatsApp images using AI (CLIP model) - GPU accelerated',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would happen
  %(prog)s "/path/to/WhatsApp Images" --dry-run
  
  # Actually move files
  %(prog)s "/path/to/WhatsApp Images"
  
  # Use larger model for better accuracy
  %(prog)s "/path/to/WhatsApp Images" --model openai/clip-vit-large-patch14
  
  # Lower confidence threshold (more aggressive classification)
  %(prog)s "/path/to/WhatsApp Images" --confidence 0.4
  
Available models:
  - openai/clip-vit-base-patch32 (default, fast, 400MB)
  - openai/clip-vit-large-patch14 (better accuracy, slower, 1.7GB)
  - google/siglip-so400m-patch14-384 (newer, good balance)
        """
    )
    
    parser.add_argument(
        'directory',
        help='Path to WhatsApp Images directory'
    )
    
    parser.add_argument(
        '--model',
        default='openai/clip-vit-base-patch32',
        help='CLIP model to use (default: openai/clip-vit-base-patch32)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.55,
        help='Minimum confidence threshold 0-1 (default: 0.55)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze only, do not move files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print classification details for each image'
    )
    
    args = parser.parse_args()
    
    # Validate confidence threshold
    if not 0 <= args.confidence <= 1:
        print("Error: Confidence threshold must be between 0 and 1")
        sys.exit(1)
    
    separate_whatsapp_images(
        source_dir=args.directory,
        model_name=args.model,
        confidence_threshold=args.confidence,
        dry_run=args.dry_run,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
