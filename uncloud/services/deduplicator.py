"""Duplicate resolution service."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from ..core.models import HashResult, DuplicateGroup
from ..core.config import DuplicatePolicy


def verify_hash_collision(path1: Path, path2: Path) -> bool:
    """Verify if two files with the same perceptual hash are actually the same.
    
    For pHash collisions, we compare file sizes and SHA256.
    
    Returns:
        True if files are genuinely the same, False if it's a false positive.
    """
    try:
        size1 = path1.stat().st_size
        size2 = path2.stat().st_size
        
        # If sizes differ significantly, probably different files
        size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0
        if size_ratio < 0.5:
            return False
        
        # Compare SHA256
        hash1 = _sha256_file(path1)
        hash2 = _sha256_file(path2)
        return hash1 == hash2
    except Exception:
        # If we can't verify, assume same to avoid duplicates
        return True


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 of file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DuplicateResolver:
    """Handles duplicate detection and resolution.
    
    Groups files by hash and applies resolution policy.
    """
    
    def __init__(self, policy: DuplicatePolicy):
        """Initialize with resolution policy.
        
        Args:
            policy: How to handle duplicates.
        """
        self._policy = policy
    
    def group_by_hash(self, results: list[HashResult]) -> list[DuplicateGroup]:
        """Group hash results by their similarity hash.
        
        Args:
            results: List of hash results.
        
        Returns:
            List of DuplicateGroup objects.
        """
        groups: dict[str, list[HashResult]] = {}
        
        for result in results:
            if not result.is_valid:
                continue
            
            hash_val = result.similarity_hash
            if hash_val is None:
                continue
            if hash_val not in groups:
                groups[hash_val] = []
            groups[hash_val].append(result)
        
        return [
            DuplicateGroup(hash_value=h, items=tuple(items))
            for h, items in groups.items()
        ]
    
    def resolve(
        self,
        group: DuplicateGroup,
        existing_in_db: Optional[HashResult] = None,
    ) -> tuple[Optional[HashResult], list[HashResult]]:
        """Resolve a duplicate group.
        
        Args:
            group: Group of files with same hash.
            existing_in_db: Existing record in database with same hash.
        
        Returns:
            (best_to_keep, list_of_duplicates)
        """
        if self._policy == DuplicatePolicy.SKIP:
            if existing_in_db:
                return None, list(group.items)
            return group.items[0], list(group.items[1:])
        
        if self._policy == DuplicatePolicy.KEEP_BOTH:
            return None, []  # Keep all, no duplicates
        
        if self._policy == DuplicatePolicy.KEEP_FIRST:
            if existing_in_db:
                return None, list(group.items)
            return group.items[0], list(group.items[1:])
        
        # KEEP_HIGHER_RESOLUTION
        candidates = list(group.items)
        if existing_in_db:
            candidates.append(existing_in_db)
        
        # Sort by resolution descending
        candidates.sort(key=lambda x: x.resolution, reverse=True)
        best = candidates[0]
        
        # If existing is best, don't copy new ones
        if existing_in_db and best == existing_in_db:
            return None, list(group.items)
        
        # Otherwise, keep best and mark others as duplicates
        duplicates = [c for c in candidates if c != best]
        return best, duplicates
    
    def should_replace_existing(
        self,
        new_result: HashResult,
        existing_path: Path,
        existing_resolution: tuple[int, int],
    ) -> bool:
        """Check if new file should replace existing.
        
        Args:
            new_result: New file's hash result.
            existing_path: Path to existing file.
            existing_resolution: Existing file's resolution.
        
        Returns:
            True if new file should replace existing.
        """
        if self._policy != DuplicatePolicy.KEEP_HIGHER_RESOLUTION:
            return False
        
        # Verify it's actually the same content
        if not verify_hash_collision(new_result.item.path, existing_path):
            return False
        
        # Compare resolutions
        existing_pixels = existing_resolution[0] * existing_resolution[1]
        new_pixels = new_result.resolution
        
        return new_pixels > existing_pixels
