"""Tests for Rich progress reporter."""
import pytest
from pathlib import Path
from io import StringIO

from uncloud.logging.rich_logger import RichProgressReporter, QuietProgressReporter
from uncloud.core.models import ProcessingStats


class TestRichProgressReporter:
    """Tests for Rich progress reporter."""
    
    @pytest.fixture
    def reporter(self):
        """Create a reporter instance."""
        return RichProgressReporter(verbose=False, quiet=False)
    
    @pytest.fixture
    def verbose_reporter(self):
        """Create a verbose reporter."""
        return RichProgressReporter(verbose=True, quiet=False)
    
    def test_create_default(self):
        """Test default creation."""
        reporter = RichProgressReporter()
        assert reporter._verbose is False
        assert reporter._quiet is False
    
    def test_create_verbose(self):
        """Test verbose creation."""
        reporter = RichProgressReporter(verbose=True)
        assert reporter._verbose is True
    
    def test_create_quiet(self):
        """Test quiet creation."""
        reporter = RichProgressReporter(quiet=True)
        assert reporter._quiet is True
    
    def test_start_and_end_phase(self, reporter):
        """Test starting and ending a phase."""
        reporter.start_phase("Testing", 100)
        assert reporter._progress is not None
        assert reporter._current_task_id is not None
        
        reporter.end_phase()
        assert reporter._progress is None
    
    def test_update_phase(self, reporter):
        """Test updating phase progress."""
        reporter.start_phase("Testing", 100)
        reporter.update_phase(50)
        reporter.end_phase()
    
    def test_advance_phase(self, reporter):
        """Test advancing phase."""
        reporter.start_phase("Testing", 100)
        reporter.advance_phase(1)
        reporter.advance_phase(5)
        reporter.end_phase()
    
    def test_info(self, reporter):
        """Test info logging (should not raise)."""
        reporter.info("Test message")
    
    def test_success(self, reporter):
        """Test success logging."""
        reporter.success("Operation completed")
    
    def test_warning(self, reporter):
        """Test warning logging."""
        reporter.warning("Something might be wrong")
    
    def test_error(self, reporter):
        """Test error logging."""
        reporter.error("Something went wrong")
    
    def test_debug_verbose(self, verbose_reporter):
        """Test debug logging in verbose mode."""
        verbose_reporter.debug("Debug info")
    
    def test_debug_not_verbose(self, reporter):
        """Test debug logging is suppressed when not verbose."""
        reporter.debug("This should not appear")
    
    def test_print_header(self, reporter):
        """Test header printing."""
        reporter.print_header("Test Header")
    
    def test_print_config(self, reporter):
        """Test config printing."""
        reporter.print_config({
            "Input": "/path/to/input",
            "Output": "/path/to/output",
            "Workers": 4,
        })
    
    def test_print_stats(self, reporter):
        """Test stats printing."""
        stats = ProcessingStats(
            total_files=100,
            copied=80,
            skipped_duplicate=15,
            errors=5,
            elapsed_seconds=120.5,
        )
        reporter.print_stats(stats)
    
    def test_context_manager(self):
        """Test using as context manager."""
        with RichProgressReporter() as reporter:
            reporter.info("Inside context")
    
    def test_quiet_mode_suppresses_info(self):
        """Test quiet mode suppresses non-essential output."""
        reporter = RichProgressReporter(quiet=True)
        
        # These should not raise
        reporter.info("Suppressed")
        reporter.success("Suppressed")
        reporter.print_header("Suppressed")


class TestQuietProgressReporter:
    """Tests for quiet progress reporter."""
    
    @pytest.fixture
    def reporter(self):
        """Create a quiet reporter."""
        return QuietProgressReporter()
    
    def test_phase_methods_no_op(self, reporter):
        """Test phase methods are no-ops."""
        reporter.start_phase("Testing", 100)
        reporter.update_phase(50)
        reporter.advance_phase(10)
        reporter.end_phase()
    
    def test_logging_methods_no_op(self, reporter):
        """Test logging methods are no-ops (except warning/error)."""
        reporter.info("Suppressed")
        reporter.success("Suppressed")
        reporter.debug("Suppressed")
    
    def test_warning_outputs(self, reporter, capsys):
        """Test warning still outputs."""
        reporter.warning("Test warning")
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
    
    def test_error_outputs(self, reporter, capsys):
        """Test error still outputs."""
        reporter.error("Test error")
        captured = capsys.readouterr()
        assert "ERROR" in captured.err
    
    def test_print_methods_no_op(self, reporter):
        """Test print methods are no-ops."""
        reporter.print_header("Suppressed")
        reporter.print_config({})
        reporter.print_stats(ProcessingStats())
    
    def test_context_manager(self):
        """Test using as context manager."""
        with QuietProgressReporter() as reporter:
            reporter.info("Inside context")
