"""Model service - centralized AI model management.

This service handles loading, caching, and lifecycle of AI models used
for image processing. Models are loaded eagerly at startup with progress
feedback to avoid user confusion during processing.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import torch

logger = logging.getLogger(__name__)


class ProgressCallback(Protocol):
    """Protocol for progress reporting during model loading."""
    def __call__(self, message: str) -> None: ...


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    model: Any
    processor: Optional[Any] = None
    device: str = "cpu"
    loaded: bool = False


class ModelService:
    """Centralized service for AI model management.
    
    This service:
    - Loads models eagerly at startup with progress feedback
    - Manages GPU/CPU device selection
    - Provides shared access to models across processors
    - Handles model lifecycle (loading, unloading)
    
    Usage:
        # At command startup
        models = ModelService(device="auto")
        models.load_models(["caption"], progress_callback=console.print)
        
        # In processors
        model, processor, device = models.get_caption_model()
    """
    
    # Supported models and their configurations
    MODEL_CONFIGS = {
        "caption": {
            "name": "Salesforce/blip-image-captioning-base",
            "type": "blip",
            "description": "BLIP image captioning model (~990MB)",
        },
        "faces": {
            "name": "facenet-pytorch",
            "type": "facenet",
            "description": "Face detection and embedding model (~500MB)",
        },
        "clip": {
            "name": "openai/clip-vit-base-patch32",
            "type": "clip",
            "description": "CLIP image embedding model (~340MB)",
        },
    }
    
    def __init__(self, device: str = "auto"):
        """Initialize model service.
        
        Args:
            device: Device to use - "auto", "cuda", "cpu", or specific like "cuda:0"
        """
        self._device = self._resolve_device(device)
        self._models: dict[str, ModelInfo] = {}
        self._loaded = False
        
        logger.debug(f"ModelService initialized with device: {self._device}")
    
    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device
    
    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_models(
        self, 
        model_keys: list[str],
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        """Load specified models with progress feedback.
        
        Args:
            model_keys: List of model keys to load (e.g., ["caption", "faces"])
            progress: Optional callback for progress messages
        """
        def log(msg: str):
            if progress:
                progress(msg)
            logger.info(msg)
        
        log(f"[bold cyan]Loading AI models on {self._device}...[/bold cyan]")
        
        for key in model_keys:
            if key not in self.MODEL_CONFIGS:
                log(f"  [yellow]⚠ Unknown model: {key}[/yellow]")
                continue
            
            if key in self._models and self._models[key].loaded:
                log(f"  [dim]• {key}: already loaded[/dim]")
                continue
            
            config = self.MODEL_CONFIGS[key]
            log(f"  [cyan]• Loading {key}...[/cyan] ({config['description']})")
            
            try:
                if config["type"] == "blip":
                    self._load_blip_model(key, config)
                elif config["type"] == "facenet":
                    self._load_facenet_model(key, config)
                elif config["type"] == "clip":
                    self._load_clip_model(key, config)
                else:
                    log(f"    [red]✗ Unknown model type: {config['type']}[/red]")
                    continue
                
                log(f"    [green]✓ {key} loaded[/green]")
                
            except Exception as e:
                log(f"    [red]✗ Failed to load {key}: {e}[/red]")
                logger.exception(f"Failed to load model {key}")
        
        self._loaded = True
        log(f"[bold green]✓ Model loading complete[/bold green]")
    
    def _load_blip_model(self, key: str, config: dict) -> None:
        """Load BLIP captioning model."""
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        model_name = config["name"]
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        model.to(self._device)
        model.eval()
        
        self._models[key] = ModelInfo(
            name=model_name,
            model=model,
            processor=processor,
            device=self._device,
            loaded=True,
        )
    
    def _load_facenet_model(self, key: str, config: dict) -> None:
        """Load FaceNet face detection/embedding model."""
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
        except ImportError:
            raise ImportError(
                "facenet-pytorch not installed. "
                "Install with: pip install facenet-pytorch"
            )
        
        # MTCNN for face detection
        mtcnn = MTCNN(
            device=self._device,
            keep_all=True,  # Return all faces, not just the largest
            min_face_size=20,
        )
        
        # InceptionResnetV1 for face embeddings
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
        resnet.to(self._device)
        
        self._models[key] = ModelInfo(
            name=config["name"],
            model=resnet,
            processor=mtcnn,  # MTCNN is the "processor" (detector)
            device=self._device,
            loaded=True,
        )
    
    def _load_clip_model(self, key: str, config: dict) -> None:
        """Load CLIP embedding model."""
        from transformers import CLIPProcessor, CLIPModel
        
        model_name = config["name"]
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        model.to(self._device)
        model.eval()
        
        self._models[key] = ModelInfo(
            name=model_name,
            model=model,
            processor=processor,
            device=self._device,
            loaded=True,
        )
    
    def get_model(self, key: str) -> Optional[ModelInfo]:
        """Get a loaded model by key.
        
        Args:
            key: Model key (e.g., "caption", "faces")
            
        Returns:
            ModelInfo if loaded, None otherwise
        """
        return self._models.get(key)
    
    def get_caption_model(self) -> tuple[Any, Any, str]:
        """Get BLIP captioning model components.
        
        Returns:
            Tuple of (processor, model, device)
            
        Raises:
            RuntimeError: If model not loaded
        """
        info = self._models.get("caption")
        if info is None or not info.loaded:
            raise RuntimeError(
                "Caption model not loaded. Call load_models(['caption']) first."
            )
        return info.processor, info.model, info.device
    
    def get_faces_model(self) -> tuple[Any, Any, str]:
        """Get FaceNet model components.
        
        Returns:
            Tuple of (mtcnn_detector, resnet_embedder, device)
            
        Raises:
            RuntimeError: If model not loaded
        """
        info = self._models.get("faces")
        if info is None or not info.loaded:
            raise RuntimeError(
                "Faces model not loaded. Call load_models(['faces']) first."
            )
        return info.processor, info.model, info.device
    
    def get_clip_model(self) -> tuple[Any, Any, str]:
        """Get CLIP embedding model components.
        
        Returns:
            Tuple of (processor, model, device)
            
        Raises:
            RuntimeError: If model not loaded
        """
        info = self._models.get("clip")
        if info is None or not info.loaded:
            raise RuntimeError(
                "CLIP model not loaded. Call load_models(['clip']) first."
            )
        return info.processor, info.model, info.device
    
    def is_loaded(self, key: str) -> bool:
        """Check if a model is loaded."""
        info = self._models.get(key)
        return info is not None and info.loaded
    
    def unload_model(self, key: str) -> None:
        """Unload a model to free memory.
        
        Args:
            key: Model key to unload
        """
        if key in self._models:
            info = self._models[key]
            del info.model
            if info.processor:
                del info.processor
            del self._models[key]
            
            # Force garbage collection
            if self._device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded model: {key}")
    
    def unload_all(self) -> None:
        """Unload all models."""
        for key in list(self._models.keys()):
            self.unload_model(key)
        self._loaded = False
    
    def __enter__(self) -> "ModelService":
        return self
    
    def __exit__(self, *args) -> None:
        self.unload_all()


# Global singleton for shared access
_model_service: Optional[ModelService] = None


def get_model_service(device: str = "auto") -> ModelService:
    """Get or create the global model service.
    
    Args:
        device: Device to use (only used on first call)
        
    Returns:
        The global ModelService instance
    """
    global _model_service
    if _model_service is None:
        _model_service = ModelService(device=device)
    return _model_service


def reset_model_service() -> None:
    """Reset the global model service (for testing)."""
    global _model_service
    if _model_service is not None:
        _model_service.unload_all()
        _model_service = None
