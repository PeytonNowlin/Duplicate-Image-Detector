"""CLIP embedding generation with GPU batching."""

import torch
import open_clip
from PIL import Image
from pathlib import Path
from typing import Generator
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif'}


class ImageEmbedder:
    """Generate CLIP embeddings for images with GPU acceleration."""
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = None):
        """
        Initialize the CLIP embedder.
        
        Args:
            model_name: CLIP model variant (ViT-B-32, ViT-L-14, etc.)
            pretrained: Pretrained weights source
            device: Device to run on (auto-detects GPU if available)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.visual.output_dim
        
    def get_image_files(self, directory: Path, recursive: bool = True) -> list[Path]:
        """
        Get all image files in a directory.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            
        Returns:
            List of image file paths
        """
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        
        files = []
        for path in directory.glob(pattern):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(path)
        
        return sorted(files)
    
    def load_image(self, path: Path) -> torch.Tensor | None:
        """
        Load and preprocess a single image.
        
        Args:
            path: Path to image file
            
        Returns:
            Preprocessed image tensor or None if loading fails
        """
        try:
            image = Image.open(path).convert("RGB")
            return self.preprocess(image)
        except Exception:
            return None
    
    def embed_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Generate embeddings for a batch of preprocessed images.
        
        Args:
            images: Batch of preprocessed image tensors
            
        Returns:
            Normalized embedding vectors
        """
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.split(':')[0] if ':' in self.device else self.device):
            # Pin memory and use non-blocking transfer for faster CPU->GPU
            if self.device.startswith("cuda") and not images.is_pinned():
                images = images.pin_memory()
            embeddings = self.model.encode_image(images.to(self.device, non_blocking=True))
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            return embeddings.cpu().numpy().astype(np.float32)
    
    def embed_images(
        self, 
        image_paths: list[Path], 
        batch_size: int = 32,
        progress_callback: callable = None
    ) -> Generator[tuple[list[Path], np.ndarray], None, None]:
        """
        Generate embeddings for a list of images in batches.
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images per batch
            progress_callback: Optional callback(processed, total, path) for progress updates
            
        Yields:
            Tuples of (valid_paths, embeddings) for each batch
        """
        batch_paths = []
        batch_tensors = []
        
        for i, path in enumerate(image_paths):
            tensor = self.load_image(path)
            
            if tensor is not None:
                batch_paths.append(path)
                batch_tensors.append(tensor)
            
            if progress_callback:
                progress_callback(i + 1, len(image_paths), path)
            
            # Process batch when full
            if len(batch_tensors) >= batch_size:
                stacked = torch.stack(batch_tensors)
                embeddings = self.embed_batch(stacked)
                yield batch_paths, embeddings
                batch_paths = []
                batch_tensors = []
        
        # Process remaining images
        if batch_tensors:
            stacked = torch.stack(batch_tensors)
            embeddings = self.embed_batch(stacked)
            yield batch_paths, embeddings

    def embed_images_parallel(
        self, 
        image_paths: list[Path], 
        batch_size: int = 64,
        num_workers: int = 4,
        progress_callback: callable = None
    ) -> Generator[tuple[list[Path], np.ndarray], None, None]:
        """
        Generate embeddings with parallel image loading for better GPU utilization.
        
        Uses ThreadPoolExecutor to load/preprocess images in parallel while
        the GPU processes the current batch.
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images per batch (default 64 for GPU)
            num_workers: Number of parallel image loading threads
            progress_callback: Optional callback(processed, total, path) for progress updates
            
        Yields:
            Tuples of (valid_paths, embeddings) for each batch
        """
        if not image_paths:
            return
        
        # Queue to hold preprocessed images ready for batching
        result_queue = Queue(maxsize=batch_size * 2)
        processed_count = [0]  # Use list for mutable counter in closure
        lock = threading.Lock()
        
        def load_worker(path: Path, index: int):
            """Load and preprocess a single image."""
            tensor = self.load_image(path)
            return (index, path, tensor)
        
        # Start loading images in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all loading tasks
            futures = {
                executor.submit(load_worker, path, i): i 
                for i, path in enumerate(image_paths)
            }
            
            batch_paths = []
            batch_tensors = []
            results_buffer = {}  # Buffer to maintain order
            next_index = 0
            
            for future in as_completed(futures):
                index, path, tensor = future.result()
                results_buffer[index] = (path, tensor)
                
                # Process results in order
                while next_index in results_buffer:
                    path, tensor = results_buffer.pop(next_index)
                    next_index += 1
                    
                    if tensor is not None:
                        batch_paths.append(path)
                        batch_tensors.append(tensor)
                    
                    with lock:
                        processed_count[0] += 1
                        if progress_callback:
                            progress_callback(processed_count[0], len(image_paths), path)
                    
                    # Process batch when full
                    if len(batch_tensors) >= batch_size:
                        stacked = torch.stack(batch_tensors)
                        embeddings = self.embed_batch(stacked)
                        yield batch_paths, embeddings
                        batch_paths = []
                        batch_tensors = []
            
            # Process remaining images
            if batch_tensors:
                stacked = torch.stack(batch_tensors)
                embeddings = self.embed_batch(stacked)
                yield batch_paths, embeddings
    
    def embed_directory(
        self, 
        directory: Path, 
        batch_size: int = 32, 
        recursive: bool = True,
        show_progress: bool = True
    ) -> tuple[list[Path], np.ndarray]:
        """
        Embed all images in a directory.
        
        Args:
            directory: Directory to scan
            batch_size: Batch size for processing
            recursive: Scan subdirectories
            show_progress: Show progress bar
            
        Returns:
            Tuple of (file_paths, embeddings_array)
        """
        image_paths = self.get_image_files(directory, recursive)
        
        if not image_paths:
            return [], np.array([])
        
        all_paths = []
        all_embeddings = []
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.fields[current_file]}"),
            ) as progress:
                task = progress.add_task(
                    "Embedding images...", 
                    total=len(image_paths),
                    current_file=""
                )
                
                def update_progress(processed, total, path):
                    progress.update(task, completed=processed, current_file=path.name[:30])
                
                for paths, embeddings in self.embed_images(image_paths, batch_size, update_progress):
                    all_paths.extend(paths)
                    all_embeddings.append(embeddings)
        else:
            for paths, embeddings in self.embed_images(image_paths, batch_size):
                all_paths.extend(paths)
                all_embeddings.append(embeddings)
        
        if all_embeddings:
            return all_paths, np.vstack(all_embeddings)
        return [], np.array([])

