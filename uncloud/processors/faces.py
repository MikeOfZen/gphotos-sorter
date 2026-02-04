"""Face detection and embedding processor.

Detects faces in images and computes embeddings using FaceNet.
Embeddings are stored in hex format for space efficiency.
"""
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from PIL import Image

from uncloud.core.protocols import FileProcessor, ProcessingContext

if TYPE_CHECKING:
    from uncloud.services.models import ModelService


logger = logging.getLogger(__name__)


def embedding_to_hex(embedding: np.ndarray) -> str:
    """Convert embedding array to hex string for storage.
    
    Args:
        embedding: Float32 numpy array (typically 512-dim)
        
    Returns:
        Hex string representation
        
    Example:
        >>> emb = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        >>> hex_str = embedding_to_hex(emb)
        >>> recovered = hex_to_embedding(hex_str)
        >>> np.allclose(emb, recovered)
        True
    """
    # Convert to bytes and then to hex
    return embedding.astype(np.float32).tobytes().hex()


def hex_to_embedding(hex_str: str, dim: int = 512) -> np.ndarray:
    """Convert hex string back to embedding array.
    
    Args:
        hex_str: Hex string from embedding_to_hex()
        dim: Expected embedding dimension
        
    Returns:
        Float32 numpy array
    """
    bytes_data = bytes.fromhex(hex_str)
    return np.frombuffer(bytes_data, dtype=np.float32).copy()


@dataclass
class FaceData:
    """Data about a detected face."""
    
    box: tuple[int, int, int, int]
    """Bounding box (x1, y1, x2, y2)."""
    
    confidence: float
    """Detection confidence 0-1."""
    
    embedding: np.ndarray
    """512-dimensional embedding vector."""
    
    embedding_hex: str
    """Hex-encoded embedding for storage."""
    
    def __post_init__(self):
        if not self.embedding_hex and self.embedding is not None:
            self.embedding_hex = embedding_to_hex(self.embedding)


@dataclass
class FaceConfig:
    """Configuration for face processor."""
    
    min_face_size: int = 20
    """Minimum face size in pixels."""
    
    confidence_threshold: float = 0.9
    """Minimum confidence for face detection."""
    
    max_faces: int = 10
    """Maximum number of faces to process per image."""
    
    return_embeddings: bool = True
    """Whether to compute embeddings (slower but needed for recognition)."""


class FaceProcessor(FileProcessor):
    """Detect faces and compute embeddings.
    
    Uses MTCNN for face detection and InceptionResnetV1 (FaceNet) for
    computing 512-dimensional face embeddings. Embeddings are stored
    in hex format for efficient storage in XMP metadata.
    
    Output format in metadata:
        uncloud:faces:1:<json>
        
    Where JSON is:
        {
            "count": 2,
            "faces": [
                {
                    "box": [x1, y1, x2, y2],
                    "confidence": 0.99,
                    "embedding": "<hex>"
                },
                ...
            ]
        }
    """
    
    def __init__(
        self,
        model_service: Optional["ModelService"] = None,
        config: FaceConfig | None = None,
    ):
        """Initialize face processor.
        
        Args:
            model_service: Optional shared model service.
            config: Configuration for face detection.
        """
        self._model_service = model_service
        self._config = config or FaceConfig()
        
        # Lazy loading fallback
        self._lazy_mtcnn = None
        self._lazy_resnet = None
        self._lazy_device = None
    
    @property
    def key(self) -> str:
        return "faces"
    
    @property
    def version(self) -> int:
        return 1
    
    @property
    def depends_on(self) -> list[str]:
        return []
    
    @property
    def config(self) -> FaceConfig:
        """Current configuration."""
        return self._config
    
    def can_process(self, ctx: ProcessingContext) -> bool:
        """Only process images with loaded PIL image."""
        if ctx.is_video:
            logger.debug(f"[faces] Skipping video: {ctx.path}")
            return False
        
        if ctx.image is None:
            logger.debug(f"[faces] No image loaded: {ctx.path}")
            return False
        
        return True
    
    def _get_model(self) -> tuple:
        """Get model components from service or load lazily."""
        if self._model_service is not None:
            if self._model_service.is_loaded("faces"):
                return self._model_service.get_faces_model()
            else:
                self._model_service.load_models(["faces"])
                return self._model_service.get_faces_model()
        
        # Lazy loading fallback
        if self._lazy_mtcnn is None:
            logger.info("[faces] Loading FaceNet models (lazy mode)...")
            
            try:
                from facenet_pytorch import MTCNN, InceptionResnetV1
            except ImportError:
                raise ImportError(
                    "facenet-pytorch not installed. "
                    "Install with: pip install facenet-pytorch"
                )
            
            self._lazy_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._lazy_mtcnn = MTCNN(
                device=self._lazy_device,
                keep_all=True,
                min_face_size=self._config.min_face_size,
            )
            
            self._lazy_resnet = InceptionResnetV1(pretrained='vggface2').eval()
            self._lazy_resnet.to(self._lazy_device)
            
            logger.info(f"[faces] FaceNet loaded on {self._lazy_device}")
        
        return self._lazy_mtcnn, self._lazy_resnet, self._lazy_device
    
    def process(self, ctx: ProcessingContext) -> dict:
        """Detect faces and compute embeddings.
        
        Args:
            ctx: Processing context with loaded image
            
        Returns:
            Dictionary with face count and face data:
            {
                "count": int,
                "faces": [
                    {"box": [...], "confidence": float, "embedding": "hex..."},
                    ...
                ]
            }
        """
        logger.debug(f"[faces] Processing: {ctx.path}")
        
        try:
            mtcnn, resnet, device = self._get_model()
            
            if ctx.image is None:
                raise ValueError("Image not loaded")
            
            # Convert to RGB if needed
            image = ctx.image
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Detect faces
            boxes, probs = mtcnn.detect(image)
            
            if boxes is None or len(boxes) == 0:
                logger.debug(f"[faces] No faces detected in {ctx.path.name}")
                return {"count": 0, "faces": []}
            
            # Filter by confidence and limit count
            faces_data = []
            face_tensors = []
            
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < self._config.confidence_threshold:
                    continue
                
                if len(faces_data) >= self._config.max_faces:
                    break
                
                # Extract face region for embedding
                x1, y1, x2, y2 = [int(c) for c in box]
                face_img = image.crop((x1, y1, x2, y2))
                
                # Resize to 160x160 for FaceNet
                face_img = face_img.resize((160, 160))
                
                # Convert to tensor
                face_tensor = torch.tensor(
                    np.array(face_img).transpose(2, 0, 1),
                    dtype=torch.float32
                ).unsqueeze(0) / 255.0
                
                # Normalize (ImageNet mean/std used by FaceNet)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                face_tensor = (face_tensor - mean) / std
                
                face_tensors.append(face_tensor)
                
                faces_data.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": float(prob),
                    "embedding": None,  # Will be filled after batch inference
                })
            
            # Compute embeddings in batch if enabled
            if self._config.return_embeddings and face_tensors:
                batch = torch.cat(face_tensors, dim=0).to(device)
                
                with torch.no_grad():
                    embeddings = resnet(batch).cpu().numpy()
                
                for i, emb in enumerate(embeddings):
                    faces_data[i]["embedding"] = embedding_to_hex(emb)
            
            logger.debug(
                f"[faces] Detected {len(faces_data)} faces in {ctx.path.name}"
            )
            
            return {"count": len(faces_data), "faces": faces_data}
            
        except Exception as e:
            logger.error(f"[faces] Failed to process {ctx.path}: {e}", exc_info=True)
            raise


# Export utilities
__all__ = [
    "FaceProcessor",
    "FaceConfig",
    "FaceData",
    "embedding_to_hex",
    "hex_to_embedding",
]
