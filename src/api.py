"""FastAPI backend for the web UI."""

import asyncio
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field
import base64
import io

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from PIL import Image
import numpy as np

from .embedder import ImageEmbedder
from .cache import EmbeddingCache
from .detector import detect_duplicates, DuplicateGroup
from .actions import generate_report


app = FastAPI(title="Duplicate Image Detector", version="1.0.0")

# Store for scan jobs
jobs: dict[str, "ScanJob"] = {}

# Global embedder instance (lazy-loaded)
_embedder: Optional[ImageEmbedder] = None


def get_embedder() -> ImageEmbedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = ImageEmbedder()
    return _embedder


@dataclass
class ScanJob:
    """Represents a scan job."""
    id: str
    directory: str
    threshold: float
    method: str = "clip"  # "clip" or "phash"
    status: str = "pending"  # pending, running, completed, failed
    progress: int = 0
    total: int = 0
    current_file: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    groups: list[DuplicateGroup] = field(default_factory=list)
    error: Optional[str] = None
    # Performance metrics (in seconds)
    time_scan_files: float = 0.0
    time_load_cache: float = 0.0
    time_embedding: float = 0.0  # Also used for hashing time in phash mode
    time_detection: float = 0.0
    cached_count: int = 0
    embedded_count: int = 0  # Also used for hashed count in phash mode
    
    def to_dict(self) -> dict:
        total_time = 0.0
        if self.started_at and self.completed_at:
            total_time = (self.completed_at - self.started_at).total_seconds()
        
        return {
            "id": self.id,
            "directory": self.directory,
            "threshold": self.threshold,
            "method": self.method,
            "status": self.status,
            "progress": self.progress,
            "total": self.total,
            "current_file": self.current_file,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "groups_count": len(self.groups),
            "duplicates_count": sum(len(g.duplicates) for g in self.groups),
            "error": self.error,
            "performance": {
                "total_time": round(total_time, 2),
                "scan_files": round(self.time_scan_files, 2),
                "load_cache": round(self.time_load_cache, 2),
                "embedding": round(self.time_embedding, 2),
                "detection": round(self.time_detection, 2),
                "cached_count": self.cached_count,
                "embedded_count": self.embedded_count,
                "images_per_second": round(self.total / self.time_embedding, 1) if self.time_embedding > 0 else 0,
                "method": self.method
            }
        }


class ScanRequest(BaseModel):
    directory: str
    threshold: float = 0.95
    recursive: bool = True
    skip_cache: bool = False
    method: str = "clip"  # "clip" or "phash"


class ActionRequest(BaseModel):
    job_id: str
    action: str  # "move" or "delete"
    group_ids: list[int] = []  # Empty means all
    destination: Optional[str] = None  # Required for move


async def run_scan(job: ScanJob, recursive: bool = True, skip_cache: bool = False):
    """Run the scan job in the background."""
    import time
    
    try:
        job.status = "running"
        job.started_at = datetime.now()
        
        embedder = get_embedder()
        
        # Phase 1: Scan for image files
        t0 = time.perf_counter()
        directory = Path(job.directory)
        image_paths = embedder.get_image_files(directory, recursive)
        job.total = len(image_paths)
        job.time_scan_files = time.perf_counter() - t0
        
        if not image_paths:
            job.status = "completed"
            job.completed_at = datetime.now()
            return
        
        # Branch based on method
        if job.method == "phash":
            await run_scan_phash(job, image_paths)
        else:
            await run_scan_clip(job, image_paths, skip_cache)
        
        job.status = "completed"
        job.completed_at = datetime.now()
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.now()


async def run_scan_clip(job: ScanJob, image_paths: list[Path], skip_cache: bool = False):
    """Run CLIP-based duplicate detection."""
    import time
    
    embedder = get_embedder()
    cache = EmbeddingCache()
    
    # Phase 2: Load cached embeddings (skip if benchmarking)
    t0 = time.perf_counter()
    if skip_cache:
        cached = {}
        paths_to_embed = image_paths
    else:
        cached = cache.get_batch(image_paths)
        paths_to_embed = [p for p in image_paths if p not in cached]
    job.time_load_cache = time.perf_counter() - t0
    job.cached_count = len(cached)
    
    all_paths = []
    all_embeddings = []
    
    # Phase 3: Generate embeddings for uncached images (parallel loading)
    t0 = time.perf_counter()
    if paths_to_embed:
        def update_progress(processed, total, path):
            job.progress = processed
            job.current_file = path.name
        
        # Use parallel image loading for better GPU utilization
        batch_size = 64 if embedder.device.startswith("cuda") else 32
        
        for batch_paths, embeddings in embedder.embed_images_parallel(
            paths_to_embed, 
            batch_size=batch_size,
            num_workers=4,
            progress_callback=update_progress
        ):
            for p, e in zip(batch_paths, embeddings):
                all_paths.append(p)
                all_embeddings.append(e)
                cache.set(p, e)
            
            await asyncio.sleep(0)
    
    job.time_embedding = time.perf_counter() - t0
    job.embedded_count = len(all_paths)
    
    # Add cached embeddings
    for path in image_paths:
        if path in cached:
            all_paths.append(path)
            all_embeddings.append(cached[path])
    
    # Phase 4: Find duplicates
    t0 = time.perf_counter()
    if all_embeddings:
        embeddings_array = np.vstack(all_embeddings)
        job.groups = detect_duplicates(all_paths, embeddings_array, threshold=job.threshold)
    job.time_detection = time.perf_counter() - t0


async def run_scan_phash(job: ScanJob, image_paths: list[Path]):
    """Run pHash-based duplicate detection."""
    import time
    from .phash import detect_duplicates_phash
    
    job.time_load_cache = 0  # No cache for pHash
    job.cached_count = 0
    
    def update_progress(processed, total, path):
        job.progress = processed
        job.current_file = path.name
    
    # Run pHash detection
    t0 = time.perf_counter()
    groups, stats = detect_duplicates_phash(
        image_paths,
        threshold=job.threshold,
        num_workers=8,
        progress_callback=update_progress
    )
    
    job.time_embedding = stats["hash_time"]  # Reuse field for hash time
    job.time_detection = stats["detect_time"]
    job.embedded_count = stats["hashed_count"]  # Reuse field for hashed count
    job.groups = groups


@app.get("/api/browse")
async def browse_folder():
    """Open a native folder picker dialog."""
    import threading
    result = {"path": None}
    
    def open_dialog():
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            root.attributes('-topmost', True)  # Bring to front
            
            folder_path = filedialog.askdirectory(
                title="Select Folder to Scan for Duplicates"
            )
            
            root.destroy()
            
            if folder_path:
                result["path"] = folder_path
        except Exception as e:
            result["error"] = str(e)
    
    # Run dialog in a separate thread to not block the event loop
    thread = threading.Thread(target=open_dialog)
    thread.start()
    thread.join(timeout=120)  # 2 minute timeout
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return {"path": result["path"]}


@app.post("/api/scan")
async def start_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    """Start a new scan job."""
    directory = Path(request.directory)
    
    if not directory.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.directory}")
    
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {request.directory}")
    
    job_id = str(uuid.uuid4())[:8]
    job = ScanJob(
        id=job_id,
        directory=str(directory.absolute()),
        threshold=request.threshold,
        method=request.method
    )
    jobs[job_id] = job
    
    background_tasks.add_task(run_scan, job, request.recursive, request.skip_cache)
    
    return {"job_id": job_id, "status": "started", "method": request.method}


@app.get("/api/scan/{job_id}")
async def get_scan_status(job_id: str):
    """Get the status of a scan job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id].to_dict()


@app.get("/api/scan/{job_id}/results")
async def get_scan_results(job_id: str):
    """Get the results of a completed scan job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job is {job.status}")
    
    return {
        "job_id": job_id,
        "groups": [g.to_dict() for g in job.groups]
    }


@app.post("/api/action")
async def execute_action(request: ActionRequest):
    """Execute an action on duplicate groups."""
    if request.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[request.job_id]
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    # Filter groups
    groups = job.groups
    if request.group_ids:
        groups = [g for g in groups if g.id in request.group_ids]
    
    if request.action == "move":
        if not request.destination:
            raise HTTPException(status_code=400, detail="Destination required for move action")
        
        from .actions import move_duplicates
        result = move_duplicates(groups, Path(request.destination), interactive=False)
        return result
    
    elif request.action == "delete":
        from .actions import delete_duplicates
        result = delete_duplicates(groups, interactive=False)
        return result
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")


@app.get("/api/thumbnail")
async def get_thumbnail(path: str, size: int = 150):
    """Get a thumbnail for an image."""
    file_path = Path(path)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        img = Image.open(file_path)
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img_format = "JPEG" if img.mode == "RGB" else "PNG"
        if img.mode == "RGBA":
            img_format = "PNG"
        elif img.mode != "RGB":
            img = img.convert("RGB")
            img_format = "JPEG"
        
        img.save(buffer, format=img_format, quality=85)
        buffer.seek(0)
        
        media_type = "image/jpeg" if img_format == "JPEG" else "image/png"
        return Response(content=buffer.getvalue(), media_type=media_type)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/report/{job_id}")
async def get_report(job_id: str, format: str = "json"):
    """Get a report for a completed scan."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    content = generate_report(job.groups, format=format)
    
    media_type = "application/json" if format == "json" else "text/csv"
    return Response(content=content, media_type=media_type)


# Serve static files
static_dir = Path(__file__).parent.parent / "web"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    """Serve the main page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Web UI not found. Run from project root."}

