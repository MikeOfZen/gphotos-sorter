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
        Classify an image as photo or meme.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (category, confidence, all_scores)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Flatten all prompts
            all_prompts = self.categories["photo"] + self.categories["meme"]
            
            # Run inference
            inputs = self.processor(
                text=all_prompts, 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
            
            # Calculate category scores
            photo_score = probs[:len(self.categories["photo"])].mean().item()
            meme_score = probs[len(self.categories["photo"]):].mean().item()
            
            # Determine category
            if meme_score > photo_score:
                category = "meme"
                confidence = meme_score
            else:
                category = "photo"
                confidence = photo_score
            
            scores = {
                "photo": photo_score,
                "meme": meme_score
            }
            
            return category, confidence, scores
            
        except Exception as e:
            print(f"Error classifying {image_path}: {e}")
            return "unknown", 0.0, {}


def separate_whatsapp_images(
    source_dir: str, 
    dry_run: bool = False,
    threshold: float = 0.5,
    model_name: str = "openai/clip-vit-base-patch32"
):
    """
    Separate WhatsApp images into memes and photos subdirectories using CLIP.
    
    Args:
        source_dir: Directory containing WhatsApp images
        dry_run: If True, only analyze without moving files
        threshold: Confidence threshold (0.0-1.0) for classification
        model_name: HuggingFace model to use
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Directory not found: {source_dir}")
        return
    
    # Initialize classifier
    classifier = ImageClassifier(model_name=model_name)
    
    # Create subdirectories
    photos_dir = source_path / "photos"
    memes_dir = source_path / "memes"
    uncertain_dir = source_path / "uncertain"
    
    if not dry_run:
        photos_dir.mkdir(exist_ok=True)
        memes_dir.mkdir(exist_ok=True)
        uncertain_dir.mkdir(exist_ok=True)
        print(f"\nCreated subdirectories:")
        print(f"  - {photos_dir}")
        print(f"  - {memes_dir}")
        print(f"  - {uncertain_dir}")
    
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
    uncertain_count = 0
    error_count = 0
    
    print("\nClassifying images using CLIP model...")
    print(f"Confidence threshold: {threshold}")
    
    results = []
    
    for img_file in tqdm(image_files, desc="Processing"):
        try:
            category, confidence, scores = classifier.classify_image(str(img_file))
            
            # Determine destination based on confidence
            if category == "unknown":
                destination = uncertain_dir / img_file.name
                category_label = "error"
                error_count += 1
            elif confidence < threshold:
                destination = uncertain_dir / img_file.name
                category_label = "uncertain"
                uncertain_count += 1
            elif category == "meme":
                destination = memes_dir / img_file.name
                category_label = "meme"
                meme_count += 1
            else:
                destination = photos_dir / img_file.name
                category_label = "photo"
                photo_count += 1
            
            results.append({
                'file': img_file.name,
                'category': category_label,
                'confidence': confidence,
                'scores': scores
            })
            
            if dry_run:
                # Just print what would happen
                tqdm.write(
                    f"[{category_label:9s}] {img_file.name:40s} "
                    f"(conf: {confidence:.2f}, photo: {scores.get('photo', 0):.2f}, "
                    f"meme: {scores.get('meme', 0):.2f})"
                )
            else:
                # Move the file
                shutil.move(str(img_file), str(destination))
                
        except Exception as e:
            error_count += 1
            tqdm.write(f"Error processing {img_file.name}: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Photos:               {photo_count:4d} ({photo_count/len(image_files)*100:.1f}%)")
    print(f"Memes:                {meme_count:4d} ({meme_count/len(image_files)*100:.1f}%)")
    print(f"Uncertain:            {uncertain_count:4d} ({uncertain_count/len(image_files)*100:.1f}%)")
    print(f"Errors:               {error_count:4d}")
    print(f"Total processed:      {len(image_files):4d}")
    
    if dry_run:
        print("\n[DRY RUN] No files were moved. Run without --dry-run to apply changes.")
    else:
        print(f"\nFiles organized into:")
        print(f"  - {photos_dir} ({photo_count} files)")
        print(f"  - {memes_dir} ({meme_count} files)")
        print(f"  - {uncertain_dir} ({uncertain_count} files)")
        print(f"\nReview the 'uncertain' folder and manually sort those images.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Separate WhatsApp images into memes and photos using CLIP ML model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would happen
  %(prog)s "/path/to/WhatsApp Images" --dry-run
  
  # Actually move files with default settings
  %(prog)s "/path/to/WhatsApp Images"
  
  # Use larger model for better accuracy
  %(prog)s "/path/to/WhatsApp Images" --model openai/clip-vit-large-patch14
  
  # Adjust confidence threshold
  %(prog)s "/path/to/WhatsApp Images" --threshold 0.6
  
  # Use CPU instead of GPU
  %(prog)s "/path/to/WhatsApp Images" --device cpu
        """
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
        type=float,
        default=0.5,
        help='Confidence threshold for classification (0.0-1.0, default: 0.5)'
    )
    parser.add_argument(
        '--model',
        default='openai/clip-vit-base-patch32',
        help='CLIP model to use (default: openai/clip-vit-base-patch32)'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use for inference (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    elif args.device == 'auto':
        device = None
    else:
        device = args.device
    
    separate_whatsapp_images(
        args.directory, 
        dry_run=args.dry_run,
        threshold=args.threshold,
        model_name=args.model
    )


if __name__ == '__main__':
    main()

    separate_whatsapp_images(
        source_dir=args.directory,
        model_name=args.model,
        confidence_threshold=args.confidence,
        dry_run=args.dry_run,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
