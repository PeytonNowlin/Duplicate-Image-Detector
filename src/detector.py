"""Faiss-based duplicate detection with clustering."""

import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class DuplicateGroup:
    """A group of duplicate images."""
    id: int
    paths: list[Path] = field(default_factory=list)
    similarity_scores: list[float] = field(default_factory=list)
    
    @property
    def original(self) -> Path:
        """The first (original) image in the group."""
        return self.paths[0] if self.paths else None
    
    @property
    def duplicates(self) -> list[Path]:
        """All duplicate images (excluding the original)."""
        return self.paths[1:] if len(self.paths) > 1 else []
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "original": str(self.original),
            "duplicates": [str(p) for p in self.duplicates],
            "similarity_scores": self.similarity_scores,
            "count": len(self.paths)
        }


class DuplicateDetector:
    """Detect duplicate images using Faiss similarity search."""
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the detector.
        
        Args:
            use_gpu: Whether to use GPU acceleration (if available)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu or faiss-gpu")
        
        self.use_gpu = use_gpu and hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0
        self.index: Optional[faiss.Index] = None
        self.paths: list[Path] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def build_index(self, paths: list[Path], embeddings: np.ndarray):
        """
        Build the Faiss index from embeddings.
        
        Args:
            paths: List of image file paths
            embeddings: Corresponding embedding vectors (normalized)
        """
        self.paths = list(paths)
        self.embeddings = embeddings.astype(np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Create index (Inner Product = cosine similarity for normalized vectors)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        
        # Move to GPU if available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            except Exception:
                pass  # Fall back to CPU
        
        self.index.add(self.embeddings)
    
    def find_duplicates(
        self, 
        threshold: float = 0.95,
        max_results_per_image: int = 100
    ) -> list[DuplicateGroup]:
        """
        Find all duplicate groups above the similarity threshold.
        
        Args:
            threshold: Minimum similarity score (0-1) to consider as duplicate
            max_results_per_image: Maximum neighbors to search per image
            
        Returns:
            List of DuplicateGroup objects
        """
        if self.index is None or len(self.paths) == 0:
            return []
        
        # Search for similar images
        k = min(max_results_per_image, len(self.paths))
        similarities, indices = self.index.search(self.embeddings, k)
        
        # Track which images have been assigned to a group
        assigned = set()
        groups = []
        group_id = 0
        
        for i in range(len(self.paths)):
            if i in assigned:
                continue
            
            # Find all duplicates for this image
            group_paths = [self.paths[i]]
            group_scores = [1.0]  # Self-similarity
            
            for j, (sim, idx) in enumerate(zip(similarities[i], indices[i])):
                if idx == i or idx in assigned:
                    continue
                if sim >= threshold:
                    group_paths.append(self.paths[idx])
                    group_scores.append(float(sim))
                    assigned.add(idx)
            
            # Only create group if there are duplicates
            if len(group_paths) > 1:
                assigned.add(i)
                groups.append(DuplicateGroup(
                    id=group_id,
                    paths=group_paths,
                    similarity_scores=group_scores
                ))
                group_id += 1
        
        return groups
    
    def find_similar(self, embedding: np.ndarray, k: int = 10, threshold: float = 0.0) -> list[tuple[Path, float]]:
        """
        Find images similar to a given embedding.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (path, similarity) tuples
        """
        if self.index is None:
            return []
        
        embedding = embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)
        
        similarities, indices = self.index.search(embedding, k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= threshold and idx < len(self.paths):
                results.append((self.paths[idx], float(sim)))
        
        return results


def detect_duplicates(
    paths: list[Path],
    embeddings: np.ndarray,
    threshold: float = 0.95,
    use_gpu: bool = True
) -> list[DuplicateGroup]:
    """
    Convenience function to detect duplicates.
    
    Args:
        paths: List of image file paths
        embeddings: Corresponding embedding vectors
        threshold: Similarity threshold
        use_gpu: Use GPU acceleration
        
    Returns:
        List of duplicate groups
    """
    detector = DuplicateDetector(use_gpu=use_gpu)
    detector.build_index(paths, embeddings)
    return detector.find_duplicates(threshold=threshold)

