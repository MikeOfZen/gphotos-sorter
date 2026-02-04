"""File processors for the processing pipeline.

Each processor extracts specific information from media files:
- Hashes
- Captions
- Faces (future)
- Objects (future)
"""

from .hash import PerceptualHashProcessor
from .caption import CaptionProcessor

__all__ = [
    'PerceptualHashProcessor',
    'CaptionProcessor',
]
