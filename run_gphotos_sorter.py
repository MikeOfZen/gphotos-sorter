"""Entry point for PyInstaller executable."""
import sys
from pathlib import Path

# Add the package to the path
package_root = Path(__file__).parent

if __name__ == "__main__":
    from gphotos_sorter.cli import main
    main()
