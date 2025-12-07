"""Perceptual hash (pHash) based duplicate detection."""

import imagehash
from PIL import Image
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .detector import DuplicateGroup


def compute_phash(path: Path, hash_size: int = 16) -> Optional[imagehash.ImageHash]:
    try:
        img = Image.open(path).convert("RGB")
        return imagehash.phash(img, hash_size=hash_size)
    except Exception:
        return None


def compute_hashes_parallel(
    image_paths: list[Path],
    hash_size: int = 16,
    num_workers: int = 8,
    progress_callback: callable = None
) -> dict[Path, imagehash.ImageHash]:
    results = {}
    processed = [0]
    
    def hash_worker(path: Path):
        return path, compute_phash(path, hash_size)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(hash_worker, p): p for p in image_paths}
        
        for future in as_completed(futures):
            path, hash_val = future.result()
            if hash_val is not None:
                results[path] = hash_val
            
            processed[0] += 1
            if progress_callback:
                progress_callback(processed[0], len(image_paths), path)
    
    return results


def find_duplicates_phash(
    paths: list[Path],
    hashes: dict[Path, imagehash.ImageHash],
    threshold: float = 0.95,
    hash_size: int = 16
) -> list[DuplicateGroup]:
    max_bits = hash_size * hash_size
    max_distance = int((1 - threshold) * max_bits)
    
    valid_paths = [p for p in paths if p in hashes]
    if not valid_paths:
        return []
    
    assigned = set()
    groups = []
    group_id = 0
    
    for i, path1 in enumerate(valid_paths):
        if path1 in assigned:
            continue
        
        hash1 = hashes[path1]
        group_paths = [path1]
        group_scores = [1.0]
        
        for path2 in valid_paths[i + 1:]:
            if path2 in assigned:
                continue
            
            hash2 = hashes[path2]
            distance = hash1 - hash2
            
            if distance <= max_distance:
                similarity = 1.0 - (distance / max_bits)
                group_paths.append(path2)
                group_scores.append(float(similarity))
                assigned.add(path2)
        
        if len(group_paths) > 1:
            assigned.add(path1)
            groups.append(DuplicateGroup(
                id=group_id,
                paths=group_paths,
                similarity_scores=group_scores
            ))
            group_id += 1
    
    return groups


def detect_duplicates_phash(
    paths: list[Path],
    threshold: float = 0.95,
    hash_size: int = 16,
    num_workers: int = 8,
    progress_callback: callable = None
) -> tuple[list[DuplicateGroup], dict]:
    import time
    
    t0 = time.perf_counter()
    hashes = compute_hashes_parallel(
        paths, hash_size=hash_size,
        num_workers=num_workers,
        progress_callback=progress_callback
    )
    hash_time = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    groups = find_duplicates_phash(paths, hashes, threshold, hash_size)
    detect_time = time.perf_counter() - t0
    
    return groups, {
        "hash_time": hash_time,
        "detect_time": detect_time,
        "hashed_count": len(hashes)
    }
