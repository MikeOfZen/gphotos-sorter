"""Image captioning processor using BLIP."""
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from uncloud.core.protocols import FileProcessor, ProcessingContext


logger = logging.getLogger(__name__)


class CaptionProcessor(FileProcessor):
    """Generate natural language descriptions of images using BLIP.
    
    Uses Salesforce BLIP-base model for image captioning. Only processes
    images (not videos). Stores caption in XMP:Subject as uncloud:caption:1:"text".
    
    Model is loaded lazily on first use and cached for subsequent images.
    Automatically uses CUDA if available, falls back to CPU.
    """
    
    # Class-level model cache to avoid reloading
    _model: Optional[BlipForConditionalGeneration] = None
    _processor: Optional[BlipProcessor] = None
    _device: Optional[str] = None
    
    @property
    def key(self) -> str:
        return "caption"
    
    @property
    def version(self) -> int:
        return 1
    
    @property
    def depends_on(self) -> list[str]:
        return []  # No dependencies
    
    def can_process(self, ctx: ProcessingContext) -> bool:
        """Only process images, not videos."""
        if ctx.is_video:
            logger.debug(f"[caption] Skipping video: {ctx.path}")
            return False
        
        if ctx.image is None:
            logger.debug(f"[caption] No image loaded: {ctx.path}")
            return False
        
        return True
    
    @classmethod
    def _ensure_model_loaded(cls) -> tuple[BlipProcessor, BlipForConditionalGeneration, str]:
        """Load BLIP model if not already loaded."""
        if cls._model is None or cls._processor is None:
            logger.info("[caption] Loading BLIP model (first time, may take a moment)...")
            
            # Determine device
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[caption] Using device: {cls._device}")
            
            # Load processor and model
            model_name = "Salesforce/blip-image-captioning-base"
            cls._processor = BlipProcessor.from_pretrained(model_name)
            cls._model = BlipForConditionalGeneration.from_pretrained(model_name)
            cls._model.to(cls._device)
            cls._model.eval()  # Set to evaluation mode
            
            logger.info(f"[caption] BLIP model loaded on {cls._device}")
        
        return cls._processor, cls._model, cls._device
    
    def process(self, ctx: ProcessingContext) -> str:
        """Generate caption for image.
        
        Args:
            ctx: Processing context with loaded image
            
        Returns:
            Natural language caption string
        """
        logger.debug(f"[caption] Processing: {ctx.path}")
        
        try:
            # Ensure model is loaded
            processor, model, device = self._ensure_model_loaded()
            
            # Prepare image
            if ctx.image is None:
                raise ValueError("Image not loaded")
            
            # Convert to RGB if needed
            image = ctx.image
            if image.mode != "RGB":
                logger.debug(f"[caption] Converting {image.mode} to RGB")
                image = image.convert("RGB")
            
            # Process image
            logger.debug(f"[caption] Running inference on {ctx.path.name}")
            inputs = processor(image, return_tensors="pt").to(device)
            
            # Generate caption
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            logger.debug(f"[caption] Generated: {caption[:80]}...")
            return caption
            
        except Exception as e:
            logger.error(f"[caption] Failed to process {ctx.path}: {e}", exc_info=True)
            raise
