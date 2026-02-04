"""Image captioning processor using BLIP or other models.

Supports configurable prompts and interchangeable models.
"""
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

import torch
from PIL import Image

from uncloud.core.protocols import FileProcessor, ProcessingContext

if TYPE_CHECKING:
    from uncloud.services.models import ModelService


logger = logging.getLogger(__name__)


# Available caption models
CaptionModel = Literal["blip-base", "blip-large", "blip2"]

# Model registry with HuggingFace names
CAPTION_MODELS = {
    "blip-base": "Salesforce/blip-image-captioning-base",
    "blip-large": "Salesforce/blip-image-captioning-large",
    "blip2": "Salesforce/blip2-opt-2.7b",
}


@dataclass
class CaptionConfig:
    """Configuration for caption processor."""
    
    model: CaptionModel = "blip-base"
    """Which model to use for captioning."""
    
    prompt: str | None = None
    """Optional prompt to guide caption generation.
    
    Examples:
        - "a photograph of"
        - "this is a picture of"
        - "describe this image:"
    
    If None, uses unconditional generation.
    """
    
    max_tokens: int = 50
    """Maximum number of tokens to generate."""
    
    min_tokens: int = 5
    """Minimum number of tokens to generate."""
    
    num_beams: int = 3
    """Number of beams for beam search (higher = better but slower)."""


class CaptionProcessor(FileProcessor):
    """Generate natural language descriptions of images.
    
    Supports multiple caption models (BLIP, BLIP-2) with configurable prompts.
    Only processes images (not videos). Stores caption in XMP:Subject.
    
    The processor can either:
    1. Use a shared ModelService (preferred - model loaded upfront)
    2. Load the model lazily on first use (fallback)
    
    Prompt examples:
        - None (default): Unconditional generation
        - "a photograph of": Guides toward photography description
        - "describe the scene:": More detailed description
        - "what objects are in this image?": Object-focused
    """
    
    def __init__(
        self, 
        model_service: Optional["ModelService"] = None,
        config: CaptionConfig | None = None,
    ):
        """Initialize caption processor.
        
        Args:
            model_service: Optional shared model service. If not provided,
                          model will be loaded lazily on first use.
            config: Configuration for captioning. Uses defaults if not provided.
        """
        self._model_service = model_service
        self._config = config or CaptionConfig()
        
        # Fallback for lazy loading if no model service provided
        self._lazy_processor = None
        self._lazy_model = None
        self._lazy_device = None
    
    @property
    def key(self) -> str:
        return "caption"
    
    @property
    def version(self) -> int:
        return 1
    
    @property
    def depends_on(self) -> list[str]:
        return []  # No dependencies
    
    @property
    def config(self) -> CaptionConfig:
        """Current configuration."""
        return self._config
    
    def can_process(self, ctx: ProcessingContext) -> bool:
        """Only process images, not videos."""
        if ctx.is_video:
            logger.debug(f"[caption] Skipping video: {ctx.path}")
            return False
        
        if ctx.image is None:
            logger.debug(f"[caption] No image loaded: {ctx.path}")
            return False
        
        return True
    
    def _get_model(self) -> tuple:
        """Get model components from service or load lazily."""
        # Prefer model service if available
        if self._model_service is not None:
            if self._model_service.is_loaded("caption"):
                return self._model_service.get_caption_model()
            else:
                # Load through service if not already loaded
                self._model_service.load_models(["caption"])
                return self._model_service.get_caption_model()
        
        # Fallback: lazy loading without service
        if self._lazy_model is None:
            model_name = CAPTION_MODELS.get(
                self._config.model, 
                CAPTION_MODELS["blip-base"]
            )
            logger.info(f"[caption] Loading {self._config.model} model (lazy mode)...")
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            self._lazy_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._lazy_processor = BlipProcessor.from_pretrained(model_name)
            self._lazy_model = BlipForConditionalGeneration.from_pretrained(model_name)
            self._lazy_model.to(self._lazy_device)
            self._lazy_model.eval()
            
            logger.info(f"[caption] {self._config.model} model loaded on {self._lazy_device}")
        
        return self._lazy_processor, self._lazy_model, self._lazy_device
    
    def process(self, ctx: ProcessingContext) -> str:
        """Generate caption for image.
        
        Args:
            ctx: Processing context with loaded image
            
        Returns:
            Natural language caption string
        """
        logger.debug(f"[caption] Processing: {ctx.path}")
        
        try:
            # Get model components
            processor, model, device = self._get_model()
            
            # Prepare image
            if ctx.image is None:
                raise ValueError("Image not loaded")
            
            # Convert to RGB if needed
            image = ctx.image
            if image.mode != "RGB":
                logger.debug(f"[caption] Converting {image.mode} to RGB")
                image = image.convert("RGB")
            
            # Process image with optional prompt
            logger.debug(f"[caption] Running inference on {ctx.path.name}")
            
            if self._config.prompt:
                # Conditional generation with prompt
                inputs = processor(
                    image, 
                    self._config.prompt,
                    return_tensors="pt"
                ).to(device)
            else:
                # Unconditional generation
                inputs = processor(image, return_tensors="pt").to(device)
            
            # Generate caption
            with torch.no_grad():
                out = model.generate(
                    **inputs, 
                    max_new_tokens=self._config.max_tokens,
                    min_length=self._config.min_tokens,
                    num_beams=self._config.num_beams,
                )
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            # Clean up caption (remove prompt if present at start)
            if self._config.prompt and caption.lower().startswith(self._config.prompt.lower()):
                caption = caption[len(self._config.prompt):].strip()
            
            logger.debug(f"[caption] Generated: {caption[:80]}...")
            return caption
            
        except Exception as e:
            logger.error(f"[caption] Failed to process {ctx.path}: {e}", exc_info=True)
            raise
