"""Rich-based progress reporter implementation."""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import (
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TaskID,
    Task,
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from ..core.models import ProcessingStats


class FilesPerSecondColumn(ProgressColumn):
    """Renders files per second as a rolling average."""
    
    def __init__(self, window_size: int = 10):
        """Initialize with rolling window size.
        
        Args:
            window_size: Number of samples for rolling average.
        """
        super().__init__()
        self._window_size = window_size
        self._samples: deque[tuple[float, int]] = deque(maxlen=window_size)
        self._last_completed = 0
        self._start_time: Optional[float] = None
    
    def render(self, task: Task) -> Text:
        """Render the speed column."""
        completed = int(task.completed)
        current_time = time.time()
        
        # Initialize on first call
        if self._start_time is None:
            self._start_time = current_time
            self._last_completed = completed
            return Text("-- f/s", style="magenta")
        
        # Record sample if progress changed
        if completed > self._last_completed:
            self._samples.append((current_time, completed))
            self._last_completed = completed
        
        # Calculate rolling average
        if len(self._samples) >= 2:
            oldest_time, oldest_completed = self._samples[0]
            newest_time, newest_completed = self._samples[-1]
            
            time_diff = newest_time - oldest_time
            completed_diff = newest_completed - oldest_completed
            
            if time_diff > 0:
                speed = completed_diff / time_diff
                return Text(f"{speed:.1f} f/s", style="magenta")
        
        # Fallback to overall average
        elapsed = current_time - self._start_time
        if elapsed > 0 and completed > 0:
            speed = completed / elapsed
            return Text(f"{speed:.1f} f/s", style="magenta")
        
        return Text("-- f/s", style="magenta")


class RichProgressReporter:
    """Progress reporter using Rich for beautiful terminal output.
    
    Implements the ProgressReporter protocol with Rich console output.
    """
    
    def __init__(
        self, 
        verbose: bool = False,
        quiet: bool = False,
    ):
        """Initialize the reporter.
        
        Args:
            verbose: Enable verbose output.
            quiet: Suppress all non-essential output.
        """
        self._console = Console(stderr=True)
        self._verbose = verbose
        self._quiet = quiet
        self._progress: Optional[Progress] = None
        self._current_task_id: Optional[TaskID] = None
        self._phase_name: str = ""
    
    # --- Phase Management ---
    
    def start_phase(self, name: str, total: int) -> None:
        """Start a new processing phase with progress bar."""
        self._phase_name = name
        
        if self._quiet:
            return
        
        # Create progress bar for this phase with files/s speed indicator
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[cyan]•"),
            FilesPerSecondColumn(),
            TextColumn("[cyan]•"),
            TimeElapsedColumn(),
            TextColumn("[cyan]•"),
            TimeRemainingColumn(),
            console=self._console,
            transient=False,
        )
        self._progress.start()
        self._current_task_id = self._progress.add_task(
            name, 
            total=total,
        )
    
    def update_phase(self, completed: int, description: Optional[str] = None) -> None:
        """Update phase progress."""
        if self._progress and self._current_task_id is not None:
            if description:
                self._progress.update(self._current_task_id, completed=completed, description=description)
            else:
                self._progress.update(self._current_task_id, completed=completed)
    
    def advance_phase(self, amount: int = 1) -> None:
        """Advance the current phase by amount."""
        if self._progress and self._current_task_id is not None:
            self._progress.advance(self._current_task_id, amount)
    
    def end_phase(self) -> None:
        """End the current phase."""
        if self._progress:
            self._progress.stop()
            self._progress = None
            self._current_task_id = None
    
    # --- Logging Methods ---
    
    def info(self, message: str) -> None:
        """Log an info message."""
        if not self._quiet:
            self._console.print(f"[blue]ℹ[/blue] {message}")
    
    def success(self, message: str) -> None:
        """Log a success message."""
        if not self._quiet:
            self._console.print(f"[green]✓[/green] {message}")
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self._console.print(f"[yellow]⚠[/yellow] {message}")
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self._console.print(f"[red]✗[/red] {message}", style="red")
    
    def debug(self, message: str) -> None:
        """Log a debug message (only in verbose mode)."""
        if self._verbose:
            self._console.print(f"[dim]  {message}[/dim]")
    
    # --- Specialized Output ---
    
    def print_header(self, title: str) -> None:
        """Print a styled header."""
        if self._quiet:
            return
        
        text = Text(title, style="bold cyan")
        self._console.print(Panel(text, border_style="cyan"))
    
    def print_config(self, config_items: dict) -> None:
        """Print configuration as a table."""
        if self._quiet:
            return
        
        table = Table(title="Configuration", show_header=True, header_style="bold")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        
        for key, value in config_items.items():
            table.add_row(key, str(value))
        
        self._console.print(table)
    
    def print_stats(self, stats: ProcessingStats) -> None:
        """Print processing statistics."""
        if self._quiet:
            return
        
        table = Table(title="Processing Complete", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")
        
        table.add_row("Files Scanned", str(stats.files_scanned))
        table.add_row("Files Copied", str(stats.files_copied))
        table.add_row("Duplicates Skipped", str(stats.duplicates_skipped))
        table.add_row("Errors", str(stats.errors))
        
        if stats.hash_collisions > 0:
            table.add_row("Hash Collisions", str(stats.hash_collisions))
        
        # Add timing info
        if stats.elapsed_seconds > 0:
            rate = stats.files_scanned / stats.elapsed_seconds
            table.add_row("", "")  # Blank row
            table.add_row("Time Elapsed", f"{stats.elapsed_seconds:.1f}s")
            table.add_row("Processing Rate", f"{rate:.1f} files/sec")
        
        self._console.print(table)
    
    def print_duplicate_group(
        self, 
        hash_val: str, 
        paths: list[Path], 
        kept: Path,
    ) -> None:
        """Print info about a duplicate group."""
        if not self._verbose:
            return
        
        self._console.print(f"\n[cyan]Duplicate group[/cyan] ({hash_val[:8]}...):")
        for path in paths:
            marker = "[green]✓ kept[/green]" if path == kept else "[dim]skip[/dim]"
            self._console.print(f"  {marker} {path}")
    
    def print_recovery_info(self, pending_count: int, action: str) -> None:
        """Print crash recovery information."""
        if pending_count == 0:
            return
        
        self._console.print(
            f"[yellow]⚠ Found {pending_count} pending operations from previous run[/yellow]"
        )
        self._console.print(f"  Action: {action}")
    
    # --- Context Managers ---
    
    def __enter__(self) -> "RichProgressReporter":
        return self
    
    def __exit__(self, *args) -> None:
        self.end_phase()


class QuietProgressReporter:
    """Minimal progress reporter that only shows errors."""
    
    def start_phase(self, name: str, total: int) -> None:
        pass
    
    def update_phase(self, completed: int, description: Optional[str] = None) -> None:
        pass
    
    def advance_phase(self, amount: int = 1) -> None:
        pass
    
    def end_phase(self) -> None:
        pass
    
    def info(self, message: str) -> None:
        pass
    
    def success(self, message: str) -> None:
        pass
    
    def warning(self, message: str) -> None:
        print(f"WARNING: {message}", file=sys.stderr)
    
    def error(self, message: str) -> None:
        print(f"ERROR: {message}", file=sys.stderr)
    
    def debug(self, message: str) -> None:
        pass
    
    def print_header(self, title: str) -> None:
        pass
    
    def print_config(self, config_items: dict) -> None:
        pass
    
    def print_stats(self, stats: ProcessingStats) -> None:
        pass
    
    def print_duplicate_group(
        self, 
        hash_val: str, 
        paths: list[Path], 
        kept: Path,
    ) -> None:
        pass
    
    def print_recovery_info(self, pending_count: int, action: str) -> None:
        pass
    
    def __enter__(self) -> "QuietProgressReporter":
        return self
    
    def __exit__(self, *args) -> None:
        pass
