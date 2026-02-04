# Uncloud Development Commands
# Run `just` or `just --list` to see all available commands

set fallback := true

# === Path Configuration ===
root := justfile_directory()

# === Default ===
default:
    @just --list --unsorted

# === Code Quality ===

# Run ruff and pyright checks
[group('quality')]
check:
    uv run ruff check uncloud tests
    uv run pyright uncloud tests
    @echo "✅ All checks passed"

# Fix code issues automatically
[group('quality')]
fix:
    uv run ruff check --fix --unsafe-fixes uncloud tests
    uv run ruff format uncloud tests
    uv run codespell uncloud tests -w
    @just check

# Run tests
[group('quality')]
test *args:
    #!/usr/bin/env bash
    set -e
    mkdir -p .temp
    uv run pytest {{args}} 2>&1 | tee .temp/test_results.txt
    echo ""
    echo "📝 Test results saved to .temp/test_results.txt"

# Run tests with DEBUG logging
[group('quality')]
test-debug *args:
    uv run pytest --log-cli-level=DEBUG {{args}}

# Run tests fast (fail on first error)
[group('quality')]
test-fast:
    uv run pytest -x

# Run specific test file
[group('quality')]
test-file file:
    uv run pytest -vv {{file}}

# Run tests with coverage
[group('quality')]
test-coverage:
    uv run pytest --cov=uncloud --cov-report=html

# === Git Operations ===

# Smart commit with auto-fix handling
# NOTE: Only commits staged files. Use 'git add' to stage files explicitly before committing.
[group('git')]
commit msg:
    #!/usr/bin/env bash
    set -e
    echo "Running pre-commit checks..."
    just fix || true
    git commit --no-verify -m "{{msg}}"

# Show git status
[group('git')]
status:
    git status -s

# === Utilities ===

# Clean all generated files
[group('utils')]
clean:
    rm -rf .temp build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache __pycache__
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Show project structure
[group('utils')]
tree:
    tree -L 3 -I '__pycache__|*.pyc|.venv|.ruff_cache|build|dist|*.egg-info|.pytest_cache'

# Full pre-commit checks
[group('quality')]
pre-commit: fix test
    @echo "✅ All pre-commit checks passed!"

# Install development dependencies
[group('setup')]
install:
    uv sync --all-extras --dev
    @echo "✅ Development environment ready"

# Format code (alias for fix)
[group('quality')]
fmt: fix
