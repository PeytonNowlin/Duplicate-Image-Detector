"""CLI entry point for the duplicate image detector."""

import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
import numpy as np

from .embedder import ImageEmbedder
from .cache import EmbeddingCache
from .detector import detect_duplicates
from .actions import display_results, generate_report, move_duplicates, delete_duplicates


console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """CLIP-based Duplicate Image Detector
    
    Fast, GPU-accelerated duplicate image detection using CLIP embeddings.
    """
    pass


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-t", "--threshold", default=0.95, type=float, help="Similarity threshold (0-1)")
@click.option("-r", "--recursive/--no-recursive", default=True, help="Scan subdirectories")
@click.option("--report", type=click.Path(path_type=Path), help="Save report to file (JSON/CSV)")
@click.option("--move-to", type=click.Path(path_type=Path), help="Move duplicates to folder")
@click.option("--delete", is_flag=True, help="Delete duplicates (interactive)")
@click.option("--dry-run", is_flag=True, help="Show what would be done without making changes")
@click.option("--no-cache", is_flag=True, help="Disable embedding cache")
@click.option("--batch-size", default=32, type=int, help="Batch size for GPU processing")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompts")
def scan(
    directory: Path,
    threshold: float,
    recursive: bool,
    report: Path,
    move_to: Path,
    delete: bool,
    dry_run: bool,
    no_cache: bool,
    batch_size: int,
    yes: bool
):
    """Scan a directory for duplicate images.
    
    Examples:
    
        dupimg scan /path/to/photos
        
        dupimg scan /path/to/photos --threshold 0.9 --report duplicates.json
        
        dupimg scan /path/to/photos --move-to ./duplicates
        
        dupimg scan /path/to/photos --delete --dry-run
    """
    console.print(Panel(
        f"[bold]Scanning:[/bold] {directory}\n"
        f"[bold]Threshold:[/bold] {threshold:.0%}\n"
        f"[bold]Recursive:[/bold] {recursive}",
        title="Duplicate Image Detector",
        border_style="blue"
    ))
    
    # Initialize embedder
    console.print("\n[bold]Loading CLIP model...[/bold]")
    embedder = ImageEmbedder()
    console.print(f"[green]Model loaded on {embedder.device}[/green]")
    
    # Get image files
    image_paths = embedder.get_image_files(directory, recursive)
    console.print(f"[cyan]Found {len(image_paths)} images[/cyan]\n")
    
    if not image_paths:
        console.print("[yellow]No images found in directory[/yellow]")
        return
    
    # Check cache
    cache = None if no_cache else EmbeddingCache()
    cached_embeddings = {}
    paths_to_embed = image_paths
    
    if cache:
        cached_embeddings = cache.get_batch(image_paths)
        paths_to_embed = [p for p in image_paths if p not in cached_embeddings]
        if cached_embeddings:
            console.print(f"[dim]Using {len(cached_embeddings)} cached embeddings[/dim]")
    
    # Generate embeddings for uncached images
    all_paths = []
    all_embeddings = []
    
    if paths_to_embed:
        console.print(f"[bold]Embedding {len(paths_to_embed)} images...[/bold]")
        paths, embeddings = embedder.embed_directory(
            directory, 
            batch_size=batch_size, 
            recursive=recursive,
            show_progress=True
        )
        
        # Update cache
        if cache and len(paths) > 0:
            cache.set_batch({p: e for p, e in zip(paths, embeddings)})
        
        # Combine with cached
        for path in image_paths:
            if path in cached_embeddings:
                all_paths.append(path)
                all_embeddings.append(cached_embeddings[path])
            elif path in paths:
                idx = paths.index(path)
                all_paths.append(path)
                all_embeddings.append(embeddings[idx])
    else:
        # All from cache
        all_paths = list(cached_embeddings.keys())
        all_embeddings = list(cached_embeddings.values())
    
    if not all_paths:
        console.print("[yellow]No valid images to process[/yellow]")
        return
    
    embeddings_array = np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    # Find duplicates
    console.print(f"\n[bold]Finding duplicates (threshold: {threshold:.0%})...[/bold]")
    groups = detect_duplicates(all_paths, embeddings_array, threshold=threshold)
    
    # Display results
    display_results(groups)
    
    # Generate report if requested
    if report:
        fmt = "csv" if str(report).endswith(".csv") else "json"
        generate_report(groups, report, format=fmt)
        console.print(f"\n[green]Report saved to {report}[/green]")
    
    # Move duplicates if requested
    if move_to and groups:
        move_duplicates(groups, move_to, dry_run=dry_run, interactive=not yes)
    
    # Delete duplicates if requested
    if delete and groups:
        delete_duplicates(groups, dry_run=dry_run, interactive=not yes)


@cli.command()
@click.option("--port", default=8000, type=int, help="Port to run server on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
def serve(port: int, host: str):
    """Start the web UI server.
    
    Example:
    
        dupimg serve --port 8000
    """
    console.print(Panel(
        f"[bold]Starting web server...[/bold]\n"
        f"URL: http://{host}:{port}",
        title="Duplicate Image Detector - Web UI",
        border_style="blue"
    ))
    
    import uvicorn
    from .api import app
    uvicorn.run(app, host=host, port=port)


@cli.command()
def cache_stats():
    """Show embedding cache statistics."""
    cache = EmbeddingCache()
    stats = cache.stats()
    console.print(Panel(
        f"[bold]Cached embeddings:[/bold] {stats['entries']}\n"
        f"[bold]Cache size:[/bold] {stats['size_mb']:.2f} MB",
        title="Cache Statistics",
        border_style="blue"
    ))


@cli.command()
@click.option("--prune", is_flag=True, help="Remove entries for deleted files")
def cache_clear(prune: bool):
    """Clear the embedding cache."""
    cache = EmbeddingCache()
    
    if prune:
        removed = cache.prune_missing()
        console.print(f"[green]Pruned {removed} stale entries[/green]")
    else:
        cache.clear()
        console.print("[green]Cache cleared[/green]")


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()

