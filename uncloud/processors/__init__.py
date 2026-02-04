"""File processors for the processing pipeline.

Each processor extracts specific information from media files:
- Hashes (perceptual and file)
- Captions (BLIP-based image descriptions)
- Faces (detection and embeddings)
- Objects (future - CLIP/YOLO)
"""

from .hash import PerceptualHashProcessor, VideoHashProcessor
from .caption import CaptionProcessor, CaptionConfig, CAPTION_MODELS
from .faces import FaceProcessor, FaceConfig, FaceData, embedding_to_hex, hex_to_embedding

__all__ = [
    # Hash processors
    'PerceptualHashProcessor',
    'VideoHashProcessor',
    # Caption processor
    'CaptionProcessor',
    'CaptionConfig',
    'CAPTION_MODELS',
    # Face processor
    'FaceProcessor',
    'FaceConfig',
    'FaceData',
    'embedding_to_hex',
    'hex_to_embedding',
]
