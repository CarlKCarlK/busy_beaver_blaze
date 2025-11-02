"""Pytest configuration for notebook testing."""

import pytest
from nbclient import NotebookClient


# Monkey-patch NotebookClient to skip cells tagged with 'skip-execution'
_original_async_execute_cell = NotebookClient.async_execute_cell


async def _patched_async_execute_cell(self, cell, cell_index, *args, **kwargs):
    """Skip cells tagged with 'skip-execution'."""
    tags = cell.get('metadata', {}).get('tags', [])
    if 'skip-execution' in tags:
        # Return without executing
        return cell
    return await _original_async_execute_cell(self, cell, cell_index, *args, **kwargs)


NotebookClient.async_execute_cell = _patched_async_execute_cell
