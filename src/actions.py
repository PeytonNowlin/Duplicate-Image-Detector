"""Report, move, and delete action handlers."""

import json
import csv
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from rich.panel import Panel

from .detector import DuplicateGroup


console = Console()


def generate_report(
    groups: list[DuplicateGroup],
    output_path: Optional[Path] = None,
    format: str = "json"
) -> str:
    """
    Generate a report of duplicate groups.
    
    Args:
        groups: List of duplicate groups
        output_path: Path to save report (if None, returns string)
        format: Report format ('json' or 'csv')
        
    Returns:
        Report content as string
    """
    if format == "json":
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "total_groups": len(groups),
            "total_duplicates": sum(len(g.duplicates) for g in groups),
            "groups": [g.to_dict() for g in groups]
        }
        content = json.dumps(report_data, indent=2)
    
    elif format == "csv":
        rows = []
        for group in groups:
            for i, (path, score) in enumerate(zip(group.paths, group.similarity_scores)):
                rows.append({
                    "group_id": group.id,
                    "is_original": i == 0,
                    "file_path": str(path),
                    "similarity_score": score
                })
        
        if output_path:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["group_id", "is_original", "file_path", "similarity_score"])
                writer.writeheader()
                writer.writerows(rows)
            return str(output_path)
        else:
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["group_id", "is_original", "file_path", "similarity_score"])
            writer.writeheader()
            writer.writerows(rows)
            content = output.getvalue()
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding='utf-8')
        return str(output_path)
    
    return content


def display_results(groups: list[DuplicateGroup], show_paths: bool = True):
    """
    Display duplicate groups in the console.
    
    Args:
        groups: List of duplicate groups
        show_paths: Whether to show full file paths
    """
    if not groups:
        console.print("[green]No duplicates found![/green]")
        return
    
    total_duplicates = sum(len(g.duplicates) for g in groups)
    
    console.print(Panel(
        f"[bold]Found {len(groups)} duplicate groups[/bold]\n"
        f"Total duplicate files: {total_duplicates}",
        title="Scan Results",
        border_style="blue"
    ))
    
    for group in groups[:50]:  # Limit display to first 50 groups
        table = Table(title=f"Group {group.id + 1}", show_header=True, header_style="bold cyan")
        table.add_column("Type", style="dim", width=10)
        table.add_column("File", overflow="fold")
        table.add_column("Similarity", justify="right", width=12)
        
        for i, (path, score) in enumerate(zip(group.paths, group.similarity_scores)):
            type_label = "[green]Original[/green]" if i == 0 else "[yellow]Duplicate[/yellow]"
            display_path = str(path) if show_paths else path.name
            table.add_row(type_label, display_path, f"{score:.2%}")
        
        console.print(table)
        console.print()
    
    if len(groups) > 50:
        console.print(f"[dim]... and {len(groups) - 50} more groups[/dim]")


def move_duplicates(
    groups: list[DuplicateGroup],
    destination: Path,
    dry_run: bool = False,
    interactive: bool = True
) -> dict:
    """
    Move duplicate files to a destination folder.
    
    Args:
        groups: List of duplicate groups
        destination: Destination folder for duplicates
        dry_run: If True, only show what would be moved
        interactive: Ask for confirmation before moving
        
    Returns:
        Dictionary with move statistics
    """
    destination = Path(destination)
    
    files_to_move = []
    for group in groups:
        for dup_path in group.duplicates:
            files_to_move.append(dup_path)
    
    if not files_to_move:
        console.print("[green]No duplicates to move![/green]")
        return {"moved": 0, "failed": 0, "skipped": 0}
    
    console.print(f"\n[bold]Files to move:[/bold] {len(files_to_move)}")
    console.print(f"[bold]Destination:[/bold] {destination}")
    
    if dry_run:
        console.print("\n[yellow]Dry run - no files will be moved[/yellow]")
        for path in files_to_move[:20]:
            console.print(f"  Would move: {path}")
        if len(files_to_move) > 20:
            console.print(f"  ... and {len(files_to_move) - 20} more")
        return {"moved": 0, "failed": 0, "skipped": len(files_to_move)}
    
    if interactive:
        if not Confirm.ask(f"\nMove {len(files_to_move)} files to {destination}?"):
            console.print("[yellow]Operation cancelled[/yellow]")
            return {"moved": 0, "failed": 0, "skipped": len(files_to_move)}
    
    destination.mkdir(parents=True, exist_ok=True)
    
    moved = 0
    failed = 0
    
    for path in files_to_move:
        try:
            dest_path = destination / path.name
            # Handle name collisions
            counter = 1
            while dest_path.exists():
                dest_path = destination / f"{path.stem}_{counter}{path.suffix}"
                counter += 1
            
            shutil.move(str(path), str(dest_path))
            moved += 1
        except Exception as e:
            console.print(f"[red]Failed to move {path}: {e}[/red]")
            failed += 1
    
    console.print(f"\n[green]Moved {moved} files[/green]")
    if failed:
        console.print(f"[red]Failed to move {failed} files[/red]")
    
    return {"moved": moved, "failed": failed, "skipped": 0}


def delete_duplicates(
    groups: list[DuplicateGroup],
    dry_run: bool = False,
    interactive: bool = True
) -> dict:
    """
    Delete duplicate files (keeps originals).
    
    Args:
        groups: List of duplicate groups
        dry_run: If True, only show what would be deleted
        interactive: Ask for confirmation before deleting
        
    Returns:
        Dictionary with deletion statistics
    """
    files_to_delete = []
    for group in groups:
        for dup_path in group.duplicates:
            files_to_delete.append(dup_path)
    
    if not files_to_delete:
        console.print("[green]No duplicates to delete![/green]")
        return {"deleted": 0, "failed": 0, "skipped": 0}
    
    # Calculate total size
    total_size = sum(p.stat().st_size for p in files_to_delete if p.exists())
    size_mb = total_size / (1024 * 1024)
    
    console.print(f"\n[bold red]Files to delete:[/bold red] {len(files_to_delete)}")
    console.print(f"[bold]Total size:[/bold] {size_mb:.2f} MB")
    
    if dry_run:
        console.print("\n[yellow]Dry run - no files will be deleted[/yellow]")
        for path in files_to_delete[:20]:
            console.print(f"  Would delete: {path}")
        if len(files_to_delete) > 20:
            console.print(f"  ... and {len(files_to_delete) - 20} more")
        return {"deleted": 0, "failed": 0, "skipped": len(files_to_delete)}
    
    if interactive:
        console.print("\n[bold red]WARNING: This action cannot be undone![/bold red]")
        if not Confirm.ask(f"Delete {len(files_to_delete)} files ({size_mb:.2f} MB)?"):
            console.print("[yellow]Operation cancelled[/yellow]")
            return {"deleted": 0, "failed": 0, "skipped": len(files_to_delete)}
    
    deleted = 0
    failed = 0
    
    for path in files_to_delete:
        try:
            path.unlink()
            deleted += 1
        except Exception as e:
            console.print(f"[red]Failed to delete {path}: {e}[/red]")
            failed += 1
    
    console.print(f"\n[green]Deleted {deleted} files ({size_mb:.2f} MB freed)[/green]")
    if failed:
        console.print(f"[red]Failed to delete {failed} files[/red]")
    
    return {"deleted": deleted, "failed": failed, "skipped": 0}

