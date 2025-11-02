"""Test that all Python code passes ruff linting."""

import subprocess
import sys


def test_ruff_check():
    """Run ruff check on the entire codebase."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Ruff found issues:\n{result.stdout}\n{result.stderr}"
