# Duplicate Image Detector

Find duplicate images in your photo library. Uses either CLIP (AI-based, catches resizes/crops/edits) or pHash (fast, for exact matches).

## Quick Start

```bash
# Install deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or cu118
pip install -r requirements.txt

# Web UI (recommended)
python -m src.cli serve --port 8000
# Open http://127.0.0.1:8000

# CLI
python -m src.cli scan "C:\Photos" --threshold 0.95
```

## Two Detection Methods

| Method | Speed | When to use |
|--------|-------|-------------|
| **CLIP** | ~77 img/s | Default. Finds duplicates even if cropped, filtered, or resized |
| **pHash** | ~585 img/s | 7x faster. Use for exact/near-exact duplicates or huge libraries |

*Benchmarked on 2,020 images with RTX 4060*

The web UI has a toggle to switch between them. For CLI, CLIP is used by default.

## How It Works

**CLIP mode**: Runs images through OpenAI's CLIP model to get semantic embeddings, then uses Faiss to find similar vectors. This catches duplicates that look different at the pixel level but are the same image (different resolution, Instagram filter, screenshot of a photo, etc).

**pHash mode**: Computes perceptual hashes and compares Hamming distance. Much faster since it's CPU-only, but only catches near-identical images.

Both modes support:
- Recursive directory scanning
- Similarity threshold (0.5-1.0)
- Move or delete duplicates
- JSON/CSV export
- Caching (CLIP only) so re-scans are instant

## CLI Usage

```bash
# Find duplicates
python -m src.cli scan "C:\Photos"

# Lower threshold = more matches (default 0.95)
python -m src.cli scan "C:\Photos" -t 0.90

# Move duplicates somewhere
python -m src.cli scan "C:\Photos" --move-to "C:\duplicates"

# Delete duplicates (keeps originals)
python -m src.cli scan "C:\Photos" --delete

# Preview without actually doing anything
python -m src.cli scan "C:\Photos" --delete --dry-run

# Export report
python -m src.cli scan "C:\Photos" --report duplicates.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-t, --threshold` | 0.95 | Similarity threshold. Higher = stricter |
| `--recursive/--no-recursive` | recursive | Include subdirectories |
| `--move-to PATH` | - | Move duplicates to folder |
| `--delete` | false | Delete duplicates |
| `--dry-run` | false | Preview only |
| `--report FILE` | - | Save JSON/CSV report |
| `--no-cache` | false | Skip embedding cache |
| `--batch-size` | 32 | GPU batch size |
| `-y` | false | Skip confirmation prompts |

## Threshold Guide

- **0.99+** — Exact duplicates (same file, different name)
- **0.95** — Near-identical (minor compression, metadata changes)
- **0.90** — Edited versions (cropped, filtered, watermarked)
- **0.85** — Similar (same subject, different angle)
- **0.80** — Related (same scene or context)

## Performance

On an RTX 4060 (8GB VRAM):

- **CLIP**: 77-150 img/s depending on batch size and image complexity
- **pHash**: 500-600 img/s (CPU-bound, no GPU needed)
- **Faiss search**: sub-second for 100k+ images
- **Cache hits**: ~10k lookups/sec

The cache stores CLIP embeddings in SQLite so you only compute them once. Re-scanning the same folder is nearly instant.

## Project Structure

```
src/
├── cli.py        # CLI commands
├── api.py        # FastAPI server
├── embedder.py   # CLIP embedding generation
├── detector.py   # Faiss similarity search
├── phash.py      # pHash detection
├── cache.py      # SQLite embedding cache
└── actions.py    # Move/delete/report handlers

web/
├── index.html
├── style.css
└── app.js
```

## Tech Stack

- **CLIP**: OpenCLIP (ViT-B-32)
- **Vector search**: Faiss
- **Backend**: FastAPI + Uvicorn
- **CLI**: Click + Rich
- **pHash**: imagehash library

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (optional but recommended for CLIP mode)
- ~4GB VRAM during CLIP processing

## License

MIT

---

**Peyton Nowlin**  
[nowlinautomation.com](https://nowlinautomation.com) • [peyton@nowlinautomation.com](mailto:peyton@nowlinautomation.com)
