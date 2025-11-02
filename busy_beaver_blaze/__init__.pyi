"""Type stubs for busy_beaver_blaze package.

This file provides type hints for the Rust-based components and Python extensions.
"""

from typing import Optional, Tuple, Union
from pathlib import Path
from PIL import Image

# Rust-exported class with Python-side extensions
class Visualizer:
    """Space-time diagram visualizer (Rust class with Python extensions).
    
    Core methods (from Rust):
        - step_for_secs(secs, early_stop=None) -> bool
        - to_png() -> bytes
        - resolution() -> Tuple[int, int]
        - step_count() -> int
    
    Extended methods (added by interactive.py):
        - run(early_stop, out_file=None) -> Tuple[Image.Image | bytes, int]
        - run_live(update_secs=0.1, early_stop=None, ...) -> None
    """
    
    def __init__(
        self,
        program: str,
        resolution: Tuple[int, int] = (1920, 1080),
        binning: bool = True,
        colors: Optional[list] = None,
    ) -> None:
        """Create a space-time diagram visualizer.
        
        Args:
            program: Turing machine specification string
            resolution: Target (width, height) in pixels
            binning: True for pixel averaging (smoother), False for sampling (faster)
            colors: Optional list of hex color strings (e.g., ["#FFFFFF", "#FF5500"]).
                   For multi-symbol machines, each symbol (1, 2, 3, ...) gets a color.
                   Symbol 0 is always background (first color). If fewer colors than symbols,
                   colors cycle.
        """
        ...
    
    # Core Rust methods
    def step_for_secs(
        self,
        secs: float,
        early_stop: Optional[int] = None,
        loops_per_check: int = 10_000,
    ) -> bool:
        """Step the machine for approximately `secs` seconds.
        
        Args:
            secs: Target runtime in seconds
            early_stop: Optional maximum step count
            loops_per_check: How often to check elapsed time
            
        Returns:
            True if machine can continue, False if halted
        """
        ...
    
    def to_png(self) -> bytes:
        """Render current space-time diagram as PNG bytes."""
        ...
    
    def resolution(self) -> Tuple[int, int]:
        """Get target resolution (width, height)."""
        ...
    
    def step_count(self) -> int:
        """Get current step count (1-based)."""
        ...
    
    # Python-side extensions (monkey-patched from interactive.py)
    def run(
        self,
        *,
        early_stop: int,
        out_file: Optional[Union[str, Path]] = None,
    ) -> Tuple[Union[Image.Image, bytes], int]:
        """Advance to `early_stop` and render frame (Python extension).
        
        Args:
            early_stop: Target step count
            out_file: Optional output file path
            
        Returns:
            (image, step_count) where image is PIL.Image if available, else bytes
        """
        ...
    
    def run_live(
        self,
        update_secs: float = 0.1,
        early_stop: Optional[int] = None,
        loops_per_check: int = 10_000,
        caption: str = "",
        show_stats: bool = True,
        update_interval: float = 0.0,
    ) -> None:
        """Interactive live visualization in notebooks (Python extension).
        
        Args:
            update_secs: Seconds to step before each render
            early_stop: Optional maximum step count
            loops_per_check: Time check frequency
            caption: Display caption
            show_stats: Show step count
            update_interval: Minimum seconds between updates
        """
        ...

class PngDataIterator:
    """Iterator for generating PNG frames at specified step counts."""
    
    def __init__(
        self,
        program: str,
        frame_steps: list[int],
        resolution: Tuple[int, int] = (1920, 1080),
        binning: bool = True,
        colors: Optional[list] = None,
        select: int = 1,
        part_count: Optional[int] = None,
    ) -> None: ...
    
    def __iter__(self) -> "PngDataIterator": ...
    def __next__(self) -> bytes: ...

def run_machine_steps(
    program: str,
    max_steps: int,
    force: Optional[str] = None,
) -> Tuple[int, int]:
    """Run Turing machine for max_steps.
    
    Args:
        program: Machine specification string
        max_steps: Maximum steps to execute
        force: Optional force mode ("asm" or "portable")
        
    Returns:
        (step_count, nonzero_count)
    """
    ...

# Constants
BB5_CHAMP: str
BB6_CONTENDER: str

# Type alias for SpaceByTimeMachine (same as Visualizer)
SpaceByTimeMachine = Visualizer
