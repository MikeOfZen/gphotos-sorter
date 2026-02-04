"""Centralized output service for Rich-based terminal output.

Provides a consistent, toggleable interface for all terminal output.
All output goes through this service to ensure proper Rich formatting.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


@dataclass
class OutputConfig:
    """Configuration for output service."""
    
    verbose: bool = False
    quiet: bool = False
    debug: bool = False
    use_color: bool = True
    use_emoji: bool = True


class OutputService:
    """Centralized output service with Rich formatting.
    
    Provides standard methods for all output types:
    - Status messages (info, success, warning, error)
    - Debug output (only in verbose mode)
    - Progress bars
    - Tables
    - Panels
    
    All output is configurable via OutputConfig.
    """
    
    def __init__(self, config: OutputConfig | None = None):
        """Initialize output service.
        
        Args:
            config: Output configuration. Uses defaults if not provided.
        """
        self._config = config or OutputConfig()
        self._console = Console(
            stderr=True,
            force_terminal=self._config.use_color,
            no_color=not self._config.use_color,
        )
        self._progress: Progress | None = None
        self._current_task: TaskID | None = None
    
    @property
    def console(self) -> Console:
        """Direct access to Rich console for advanced use."""
        return self._console
    
    @property
    def config(self) -> OutputConfig:
        """Current output configuration."""
        return self._config
    
    # =========================================================================
    # Status Messages
    # =========================================================================
    
    def info(self, message: str, *, prefix: str = "ℹ") -> None:
        """Print info message.
        
        Args:
            message: Message to display
            prefix: Icon prefix (default: ℹ)
        """
        if self._config.quiet:
            return
        prefix = prefix if self._config.use_emoji else "i"
        self._console.print(f"[blue]{prefix}[/blue] {message}")
    
    def success(self, message: str, *, prefix: str = "✓") -> None:
        """Print success message.
        
        Args:
            message: Message to display
            prefix: Icon prefix (default: ✓)
        """
        if self._config.quiet:
            return
        prefix = prefix if self._config.use_emoji else "*"
        self._console.print(f"[green]{prefix}[/green] {message}")
    
    def warning(self, message: str, *, prefix: str = "⚠") -> None:
        """Print warning message. Always shown, even in quiet mode.
        
        Args:
            message: Message to display
            prefix: Icon prefix (default: ⚠)
        """
        prefix = prefix if self._config.use_emoji else "!"
        self._console.print(f"[yellow]{prefix}[/yellow] {message}")
    
    def error(self, message: str, *, prefix: str = "✗") -> None:
        """Print error message. Always shown, even in quiet mode.
        
        Args:
            message: Message to display
            prefix: Icon prefix (default: ✗)
        """
        prefix = prefix if self._config.use_emoji else "X"
        self._console.print(f"[red]{prefix}[/red] {message}", style="red")
    
    def debug(self, message: str) -> None:
        """Print debug message. Only shown in verbose mode.
        
        Args:
            message: Message to display
        """
        if self._config.verbose or self._config.debug:
            self._console.print(f"[dim]  {message}[/dim]")
    
    def print(self, message: str = "", **kwargs: Any) -> None:
        """Print arbitrary message. Respects quiet mode.
        
        Args:
            message: Message to display
            **kwargs: Passed to Rich console.print()
        """
        if not self._config.quiet:
            self._console.print(message, **kwargs)
    
    # =========================================================================
    # Bullets and Lists
    # =========================================================================
    
    def bullet(self, message: str, *, level: int = 0, style: str = "cyan") -> None:
        """Print bulleted item.
        
        Args:
            message: Item text
            level: Indentation level (0, 1, 2...)
            style: Color style for bullet
        """
        if self._config.quiet:
            return
        indent = "  " * level
        bullet = "•" if self._config.use_emoji else "-"
        self._console.print(f"{indent}[{style}]{bullet}[/{style}] {message}")
    
    def step(self, message: str, *, step: int | None = None) -> None:
        """Print numbered step.
        
        Args:
            message: Step description
            step: Step number (optional)
        """
        if self._config.quiet:
            return
        if step is not None:
            self._console.print(f"[cyan]{step}.[/cyan] {message}")
        else:
            self._console.print(f"  [dim]→[/dim] {message}")
    
    # =========================================================================
    # Progress Bars
    # =========================================================================
    
    @contextmanager
    def progress(
        self,
        description: str,
        total: int,
        *,
        transient: bool = False,
    ) -> Iterator[Callable[[int], None]]:
        """Context manager for progress bar.
        
        Args:
            description: Task description
            total: Total number of items
            transient: Whether to remove progress bar when done
            
        Yields:
            Callable to advance progress by N items
            
        Example:
            with output.progress("Processing", 100) as advance:
                for item in items:
                    process(item)
                    advance(1)
        """
        if self._config.quiet:
            # Yield a no-op function
            yield lambda n: None
            return
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[cyan]•"),
            TimeElapsedColumn(),
            TextColumn("[cyan]•"),
            TimeRemainingColumn(),
            console=self._console,
            transient=transient,
        )
        
        with progress:
            task_id = progress.add_task(description, total=total)
            
            def advance(n: int = 1) -> None:
                progress.advance(task_id, n)
            
            yield advance
    
    def start_progress(self, description: str, total: int) -> None:
        """Start a long-running progress bar (non-context-manager).
        
        Use advance_progress() and end_progress() to control.
        
        Args:
            description: Task description
            total: Total number of items
        """
        if self._config.quiet:
            return
        
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[cyan]•"),
            TimeElapsedColumn(),
            TextColumn("[cyan]•"),
            TimeRemainingColumn(),
            console=self._console,
            transient=False,
        )
        self._progress.start()
        self._current_task = self._progress.add_task(description, total=total)
    
    def advance_progress(self, n: int = 1) -> None:
        """Advance current progress bar.
        
        Args:
            n: Number of items to advance
        """
        if self._progress and self._current_task is not None:
            self._progress.advance(self._current_task, n)
    
    def end_progress(self) -> None:
        """End current progress bar."""
        if self._progress:
            self._progress.stop()
            self._progress = None
            self._current_task = None
    
    # =========================================================================
    # Tables
    # =========================================================================
    
    def table(
        self,
        data: list[dict[str, Any]],
        *,
        title: str | None = None,
        columns: list[str] | None = None,
    ) -> None:
        """Print data as a table.
        
        Args:
            data: List of dictionaries with row data
            title: Optional table title
            columns: Column names (inferred from data if not provided)
        """
        if self._config.quiet or not data:
            return
        
        # Infer columns from first row if not provided
        if columns is None:
            columns = list(data[0].keys())
        
        table = Table(title=title, show_header=True, header_style="bold cyan")
        for col in columns:
            table.add_column(col)
        
        for row in data:
            table.add_row(*[str(row.get(col, "")) for col in columns])
        
        self._console.print(table)
    
    def key_value_table(
        self,
        data: dict[str, Any],
        *,
        title: str | None = None,
        key_style: str = "cyan",
        value_style: str = "white",
    ) -> None:
        """Print key-value pairs as a table.
        
        Args:
            data: Dictionary of key-value pairs
            title: Optional table title
            key_style: Style for keys
            value_style: Style for values
        """
        if self._config.quiet or not data:
            return
        
        table = Table(title=title, show_header=False, box=None)
        table.add_column("Key", style=key_style)
        table.add_column("Value", style=value_style)
        
        for key, value in data.items():
            table.add_row(str(key), str(value))
        
        self._console.print(table)
    
    # =========================================================================
    # Panels and Headers
    # =========================================================================
    
    def header(self, title: str, *, style: str = "bold cyan") -> None:
        """Print a styled header.
        
        Args:
            title: Header text
            style: Rich style for the header
        """
        if self._config.quiet:
            return
        
        text = Text(title, style=style)
        self._console.print(Panel(text, border_style="cyan"))
    
    def section(self, title: str) -> None:
        """Print a section divider.
        
        Args:
            title: Section title
        """
        if self._config.quiet:
            return
        
        self._console.print()
        self._console.print(f"[bold cyan]═══ {title} ═══[/bold cyan]")
        self._console.print()
    
    def rule(self, title: str = "") -> None:
        """Print a horizontal rule.
        
        Args:
            title: Optional title in the rule
        """
        if self._config.quiet:
            return
        
        self._console.rule(title)
    
    # =========================================================================
    # Stats and Results
    # =========================================================================
    
    def stats(
        self,
        data: dict[str, int | float | str],
        *,
        title: str = "Statistics",
    ) -> None:
        """Print statistics in a formatted table.
        
        Args:
            data: Dictionary of stat names to values
            title: Table title
        """
        if self._config.quiet or not data:
            return
        
        table = Table(title=title, show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        for key, value in data.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.2f}")
            else:
                table.add_row(key, str(value))
        
        self._console.print(table)
    
    def result(
        self,
        success: bool,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Print a result summary.
        
        Args:
            success: Whether operation succeeded
            message: Result message
            details: Optional details to display
        """
        if success:
            self.success(message)
        else:
            self.error(message)
        
        if details and not self._config.quiet:
            for key, value in details.items():
                self.bullet(f"{key}: {value}", level=1)


# Module-level singleton
_output_service: OutputService | None = None


def get_output_service() -> OutputService:
    """Get the global output service singleton.
    
    Creates a default instance if not configured.
    """
    global _output_service
    if _output_service is None:
        _output_service = OutputService()
    return _output_service


def configure_output(config: OutputConfig) -> OutputService:
    """Configure the global output service.
    
    Args:
        config: Output configuration
        
    Returns:
        Configured output service
    """
    global _output_service
    _output_service = OutputService(config)
    return _output_service


def reset_output_service() -> None:
    """Reset the global output service."""
    global _output_service
    _output_service = None
