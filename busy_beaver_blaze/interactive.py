"""Interactive visualization support for Jupyter notebooks.

This module provides live-updating displays for Turing machine visualizations
in Jupyter/VSCode notebooks, mirroring the WebAssembly interface for interactive
exploration without predetermined endpoints.
"""

from typing import Optional
import time

try:
    from IPython.display import display, Image as IPImage, clear_output
except ImportError as e:
    raise ImportError(
        "IPython is required for interactive visualization. "
        "This module is intended for use in Jupyter notebooks."
    ) from e

try:
    from .frames import resize_png
except ImportError:
    # frames.py requires PIL - provide a no-op fallback
    resize_png = None


class LiveVisualizer:
    """Live-updating visualization for Turing machines in notebooks.
    
    This class provides an interactive display similar to the WebAssembly interface,
    with frames updating in real-time as the machine runs. The user can stop
    execution with Ctrl+C (KeyboardInterrupt) or let it run until halted/early_stop.
    
    Unlike PngDataIterator, this doesn't require pre-specified frame steps - it
    mirrors the JavaScript API's `step_for_secs()` approach for indefinite running.
    
    Example:
        >>> from busy_beaver_blaze import SpaceByTimeMachine, BB5_CHAMP
        >>> from busy_beaver_blaze.interactive import LiveVisualizer
        >>> 
        >>> # Create machine (can run indefinitely)
        >>> machine = SpaceByTimeMachine(
        ...     program=BB5_CHAMP,
        ...     resolution=(1920, 1080),
        ...     binning=True
        ... )
        >>> 
        >>> # Display live updates (stop with Ctrl+C)
        >>> viz = LiveVisualizer()
        >>> viz.run(machine, run_for_secs=0.1, caption="BB5 Champion")
    """
    
    def __init__(self):
        """Initialize the live visualizer."""
        self._frames_shown = 0
    
    def run(
        self,
        machine,  # SpaceByTimeMachine instance
        run_for_secs: float = 0.1,
        early_stop: Optional[int] = None,
        loops_per_check: int = 10_000,
        caption: str = "",
        show_stats: bool = True,
        update_interval: float = 0.0,  # Min seconds between display updates
    ) -> None:
        """Run the visualization with live updates.
        
        Args:
            machine: SpaceByTimeMachine instance to visualize
            run_for_secs: How long to step before rendering (e.g., 0.1 seconds)
            early_stop: Optional maximum step count to halt at
            loops_per_check: How often to check time during stepping
            caption: Optional caption text to display
            show_stats: Whether to show step count statistics
            update_interval: Minimum seconds between display updates (0 = every render)
            
        The visualization will update in-place until the machine halts, reaches
        early_stop, or the user interrupts with Ctrl+C.
        
        Example:
            >>> # Run until 1M steps or halted
        >>> visualize_live(machine, early_stop=1_000_000, caption="Testing")
            
            >>> # Run indefinitely (stop with Ctrl+C)
            >>> visualize_live(machine, caption="Forever")
        """
        # Get target resolution from machine
        target_width, target_height = machine.resolution()
        
        last_update = time.time()
        
        try:
            while True:
                # Step the machine
                can_continue = machine.step_for_secs(run_for_secs, early_stop, loops_per_check)
                
                # Check if enough time has passed for display update
                now = time.time()
                if now - last_update >= update_interval:
                    # Render and resize to exact target
                    png_bytes = machine.to_png()
                    if resize_png is not None:
                        png_bytes = resize_png(png_bytes, (target_width, target_height))
                    
                    step_count = machine.step_count()
                    
                    self._update_display(
                        png_bytes,
                        step_count,
                        caption,
                        show_stats,
                        machine.count_nonblanks(),
                        machine.is_halted()
                    )
                    self._frames_shown += 1
                    last_update = now
                
                # Check if we should stop
                if not can_continue:
                    # Final update
                    png_bytes = machine.to_png()
                    if resize_png is not None:
                        png_bytes = resize_png(png_bytes, (target_width, target_height))
                    step_count = machine.step_count()
                    self._update_display(
                        png_bytes,
                        step_count,
                        caption,
                        show_stats,
                        machine.count_nonblanks(),
                        machine.is_halted()
                    )
                    self._frames_shown += 1
                    
                    if show_stats:
                        if machine.is_halted():
                            print(f"\n✓ Halted at step {step_count:,}")
                        elif early_stop and step_count >= early_stop:
                            print(f"\n✓ Reached early_stop at step {step_count:,}")
                    break
        
        except KeyboardInterrupt:
            # Graceful stop on Ctrl+C - show final frame
            png_bytes = machine.to_png()
            if resize_png is not None:
                png_bytes = resize_png(png_bytes, (target_width, target_height))
            step_count = machine.step_count()
            self._update_display(
                png_bytes,
                step_count,
                caption,
                show_stats,
                machine.count_nonblanks(),
                machine.is_halted()
            )
            if show_stats:
                print(f"\n⏹ Stopped by user at step {step_count:,}")
    
    def _update_display(
        self,
        png_bytes: bytes,
        step_count: int,
        caption: str,
        show_stats: bool,
        nonblanks: int,
        is_halted: bool,
    ) -> None:
        """Update the display with a new frame."""
        # Clear previous output for clean updates
        clear_output(wait=True)
        
        # Show statistics if requested
        if show_stats:
            status = " [HALTED]" if is_halted else ""
            if caption:
                print(f"Step {step_count:,} | Nonblanks: {nonblanks:,} | {caption}{status}")
            else:
                print(f"Step {step_count:,} | Nonblanks: {nonblanks:,}{status}")
        
        # Display the image
        display(IPImage(data=png_bytes, format='png'))


def visualize_live(
    machine,  # SpaceByTimeMachine instance
    run_for_secs: float = 0.1,
    early_stop: Optional[int] = None,
    loops_per_check: int = 10_000,
    caption: str = "",
    show_stats: bool = True,
    update_interval: float = 0.0,
) -> None:
    """Convenience function for live visualization.
    
    This is a simple wrapper around LiveVisualizer for quick interactive use,
    mirroring the JavaScript/WASM API for indefinite running.
    
    Args:
        machine: SpaceByTimeMachine instance to visualize
        run_for_secs: How long to step before rendering (e.g., 0.1 seconds)
        early_stop: Optional maximum step count to halt at
        loops_per_check: How often to check time during stepping
        caption: Optional caption text to display
        show_stats: Whether to show step count statistics
        update_interval: Minimum seconds between display updates
        
    Example:
        >>> from busy_beaver_blaze import SpaceByTimeMachine, BB5_CHAMP
        >>> from busy_beaver_blaze.interactive import visualize_live
        >>> 
        >>> # Create machine
        >>> machine = SpaceByTimeMachine(
        ...     program=BB5_CHAMP,
        ...     resolution=(800, 600),
        ...     binning=True
        ... )
        >>> 
        >>> # Visualize live (stop with Ctrl+C or at 1M steps)
        >>> visualize_live(machine, early_stop=1_000_000, caption="BB5 Champion")
    """
    viz = LiveVisualizer()
    viz.run(machine, run_for_secs, early_stop, loops_per_check, caption, show_stats, update_interval)

