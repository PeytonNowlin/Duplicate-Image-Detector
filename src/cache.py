"""SQLite-based embedding cache for incremental scans."""

import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional
import hashlib


class EmbeddingCache:
    """Cache embeddings in SQLite to avoid re-processing unchanged images."""
    
    def __init__(self, cache_path: Path = None):
        """
        Initialize the embedding cache.
        
        Args:
            cache_path: Path to SQLite database file. Defaults to ~/.cache/dupimg/cache.db
        """
        if cache_path is None:
            cache_path = Path.home() / ".cache" / "dupimg" / "cache.db"
        
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    path_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_mtime REAL NOT NULL,
                    file_size INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON embeddings(file_path)")
            conn.commit()
    
    @staticmethod
    def _path_hash(path: Path) -> str:
        """Generate a hash for a file path."""
        return hashlib.sha256(str(path.absolute()).encode()).hexdigest()[:32]
    
    def get(self, path: Path, model_name: str = "ViT-B-32") -> Optional[np.ndarray]:
        """
        Get cached embedding for a file if it exists and is up-to-date.
        
        Args:
            path: Path to the image file
            model_name: Model name used for embedding
            
        Returns:
            Cached embedding array or None if not found/outdated
        """
        path = Path(path)
        if not path.exists():
            return None
        
        path_hash = self._path_hash(path)
        current_mtime = path.stat().st_mtime
        current_size = path.stat().st_size
        
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute(
                """SELECT embedding, file_mtime, file_size FROM embeddings 
                   WHERE path_hash = ? AND model_name = ?""",
                (path_hash, model_name)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            embedding_blob, cached_mtime, cached_size = row
            
            # Check if file has been modified
            if cached_mtime != current_mtime or cached_size != current_size:
                return None
            
            return np.frombuffer(embedding_blob, dtype=np.float32)
    
    def set(self, path: Path, embedding: np.ndarray, model_name: str = "ViT-B-32"):
        """
        Store an embedding in the cache.
        
        Args:
            path: Path to the image file
            embedding: Embedding vector
            model_name: Model name used for embedding
        """
        path = Path(path)
        path_hash = self._path_hash(path)
        
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO embeddings 
                   (path_hash, file_path, file_mtime, file_size, embedding, model_name)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    path_hash,
                    str(path.absolute()),
                    path.stat().st_mtime,
                    path.stat().st_size,
                    embedding.astype(np.float32).tobytes(),
                    model_name
                )
            )
            conn.commit()
    
    def get_batch(self, paths: list[Path], model_name: str = "ViT-B-32") -> dict[Path, np.ndarray]:
        """
        Get cached embeddings for multiple files.
        
        Args:
            paths: List of file paths
            model_name: Model name used for embedding
            
        Returns:
            Dictionary mapping paths to their cached embeddings
        """
        result = {}
        for path in paths:
            embedding = self.get(path, model_name)
            if embedding is not None:
                result[path] = embedding
        return result
    
    def set_batch(self, path_embeddings: dict[Path, np.ndarray], model_name: str = "ViT-B-32"):
        """
        Store multiple embeddings in the cache.
        
        Args:
            path_embeddings: Dictionary mapping paths to embeddings
            model_name: Model name used for embedding
        """
        with sqlite3.connect(self.cache_path) as conn:
            for path, embedding in path_embeddings.items():
                path = Path(path)
                path_hash = self._path_hash(path)
                conn.execute(
                    """INSERT OR REPLACE INTO embeddings 
                       (path_hash, file_path, file_mtime, file_size, embedding, model_name)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        path_hash,
                        str(path.absolute()),
                        path.stat().st_mtime,
                        path.stat().st_size,
                        embedding.astype(np.float32).tobytes(),
                        model_name
                    )
                )
            conn.commit()
    
    def clear(self):
        """Clear all cached embeddings."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("DELETE FROM embeddings")
            conn.commit()
    
    def prune_missing(self) -> int:
        """
        Remove cache entries for files that no longer exist.
        
        Returns:
            Number of entries removed
        """
        removed = 0
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute("SELECT path_hash, file_path FROM embeddings")
            to_delete = []
            
            for path_hash, file_path in cursor:
                if not Path(file_path).exists():
                    to_delete.append(path_hash)
            
            for path_hash in to_delete:
                conn.execute("DELETE FROM embeddings WHERE path_hash = ?", (path_hash,))
                removed += 1
            
            conn.commit()
        
        return removed
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute("SELECT COUNT(*), SUM(LENGTH(embedding)) FROM embeddings")
            count, total_bytes = cursor.fetchone()
            return {
                "entries": count or 0,
                "size_mb": (total_bytes or 0) / (1024 * 1024)
            }

