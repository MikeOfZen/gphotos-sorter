#!/usr/bin/env python3
"""Test script to run caption processor on sample images using new service layer."""
import sys
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

from uncloud.processors.caption import CaptionProcessor
from uncloud.processors.hash import PerceptualHashProcessor
from uncloud.services import (
    AppContext, AppConfig, 
    ProcessingPipeline, PipelineStats,
)
from uncloud.persistence.database import MediaRecord
from uncloud.logging.rich_logger import RichProgressReporter

console = Console()

def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: python test_captions.py <directory> [max_files][/red]")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    if not directory.exists():
        console.print(f"[red]Directory not found: {directory}[/red]")
        sys.exit(1)
    
    # Find image files
    console.print(f"[cyan]Scanning {directory} for images...[/cyan]")
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.heic']:
        image_files.extend(directory.rglob(f"*{ext}"))
        image_files.extend(directory.rglob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)[:max_files]
    console.print(f"[green]Found {len(image_files)} images to process[/green]\n")
    
    if not image_files:
        console.print("[yellow]No images found![/yellow]")
        sys.exit(0)
    
    # Configure application context
    config = AppConfig(
        db_path=directory / ".uncloud_test.db",
        models=["caption"],  # Load caption model upfront!
        device="auto",
        write_cache=True,
        verbose=True,
    )
    
    # Use context manager for automatic cleanup
    with AppContext(config) as ctx:
        # Initialize with progress feedback
        ctx.initialize(console)
        
        # Create processors (using shared model service)
        processors = [
            PerceptualHashProcessor(),
            CaptionProcessor(model_service=ctx.models),
        ]
        
        # Create progress reporter
        progress_reporter = RichProgressReporter(verbose=True)
        
        # Create pipeline
        pipeline = ProcessingPipeline(
            processors=processors,
            metadata_service=ctx.metadata,
            progress=progress_reporter,
            write_cache=True,
        )
        
        # Process files
        stats = PipelineStats(total_files=len(image_files))
        
        console.print(f"[bold cyan]Processing {len(image_files)} images...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Processing images...", total=len(image_files))
            
            for image_path in image_files:
                try:
                    results = pipeline.process_file(image_path, stats)
                    
                    # Store in database
                    if ctx.db and ('phash' in results or 'caption' in results):
                        record = MediaRecord(
                            canonical_path=str(image_path),
                            similarity_hash=results.get('phash', ''),
                            owner="test",
                            ai_desc=results.get('caption', ''),
                        )
                        ctx.db.upsert(record)
                    
                    # Log caption
                    if 'caption' in results:
                        console.print(f"[dim]{image_path.name}:[/dim] [green]{results['caption']}[/green]")
                    
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    console.print(f"[red]Error processing {image_path.name}: {e}[/red]")
                    progress.update(task, advance=1)
        
        # Print stats
        console.print(f"\n[bold cyan]Pipeline Statistics:[/bold cyan]")
        for proc_key, proc_stats in stats.processor_stats.items():
            console.print(f"  [yellow]{proc_key}:[/yellow]")
            console.print(f"    From cache: {proc_stats['from_cache']}")
            console.print(f"    Computed: {proc_stats['computed']}")
            console.print(f"    Written: {proc_stats['written']}")
        
        # Verify metadata in a sample file
        console.print(f"\n[bold cyan]Verifying metadata in {image_files[0].name}...[/bold cyan]")
        subjects = ctx.exiftool.read_subjects(image_files[0])
        console.print(f"[dim]XMP:Subject tags:[/dim]")
        for subject in subjects:
            if subject.startswith("uncloud:"):
                console.print(f"  [green]{subject}[/green]")
        
        # Verify database
        if ctx.db:
            console.print(f"\n[bold cyan]Verifying database...[/bold cyan]")
            total_records = ctx.db.count_all()
            console.print(f"  Files in DB: {total_records}")
            if total_records > 0:
                sample = ctx.db.get_by_path(str(image_files[0]))
                if sample:
                    console.print(f"  Sample: {sample.canonical_path}")
                    console.print(f"  Hash: {sample.similarity_hash}")
                    console.print(f"  Caption: {sample.ai_desc[:100]}..." if sample.ai_desc else "  Caption: (none)")
    
    # Cleanup happens automatically via context manager
    console.print(f"\n[bold green]✓ Test complete![/bold green]")

if __name__ == "__main__":
    main()
