# cmk is this the right location for this example?
"""Generate Turing machine visualization frames using Python + Rust.

This script demonstrates using the PyO3 bindings to generate high-quality
visualization frames. It's a simplified Python version of examples/movie_list.rs
that focuses on a single Turing machine rather than batch processing.

Example usage:
    python examples/movie.py
"""

import time
from pathlib import Path

from busy_beaver_blaze import (
    PngDataIterator,
    BB5_CHAMP,      # noqa: F401
    BB6_CONTENDER,  # noqa: F401
    log_step_iterator,
    create_frame,
    RESOLUTION_2K,  # noqa: F401
    RESOLUTION_4K,  # noqa: F401
)


def create_sequential_subdir(top_dir: Path) -> tuple[Path, int]:
    """Create a new numeric subdirectory under top_dir.
    
    Finds the highest numbered directory and creates the next one.
    Returns the new path and the number.
    
    Example:
        If top_dir contains [1, 2, 5], creates directory 6.
    """
    top_dir.mkdir(parents=True, exist_ok=True)
    
    max_num = 0
    for entry in top_dir.iterdir():
        if entry.is_dir() and entry.name.isdigit():
            max_num = max(max_num, int(entry.name))
    
    new_dir_num = max_num + 1
    new_dir_path = top_dir / str(new_dir_num)
    new_dir_path.mkdir(parents=True, exist_ok=True)
    
    return new_dir_path, new_dir_num


def main():
    """Generate visualization frames for a Turing machine."""
    start_time = time.time()
    
    # Configuration
    
    program = BB6_CONTENDER
    program_name = "bb6_contender"  # Used for file naming
    resolution = RESOLUTION_4K
    early_stop = 10_000_000_000_002
    frame_count = 1000
    part_count = 32  # Number of parallel workers (None = auto-detect CPU count)
    caption = ""

    # program = BB5_CHAMP
    # program_name = "bb5_champ"  # Used for file naming
    # resolution = RESOLUTION_4K
    # early_stop = 47_176_870  # BB5 halts at this step
    # frame_count = 100
    # part_count = 8  # Number of parallel workers (None = auto-detect CPU count)
    # caption = ""

    top_directory = Path(r"m:\deldir\bbpy") / program_name    
    output_dir, run_id = create_sequential_subdir(top_directory)
    print(f"Output directory: {output_dir}")
    print(f"Run ID: {run_id}")
    
    # Generate logarithmically-spaced step indices
    print(f"Generating {frame_count} frames up to step {early_stop:,}")
    frame_steps = log_step_iterator(early_stop, frame_count)
    print(f"Frame steps: {frame_steps[:5]}...{frame_steps[-5:]}")
    
    # Create iterator
    print(f"Creating PngDataIterator with {part_count} workers...")
    iterator = PngDataIterator(
        frame_steps,
        resolution=resolution,
        early_stop=early_stop,
        program=program,
        part_count=part_count,
    )
    
    print("Processing frames...")
    frame_index = 0
    
    for step_index, png_bytes in iterator:
        elapsed = time.time() - start_time
        print(f"Frame {frame_index:04d}, Step {step_index + 1:,}, {elapsed:.1f}s elapsed")
        
        # Create frame with text overlay
        frame = create_frame(
            png_bytes,
            caption,
            step_index,
            resolution,
        )
        
        # Save frame with program name and run ID
        frame_path = output_dir / f"{program_name}_{run_id}_{frame_index:04d}.png"
        frame.save(frame_path)
        
        frame_index += 1
    elapsed = time.time() - start_time
    print(f"\nCompleted {frame_index} frames in {elapsed:.1f}s")
    print(f"Output saved to: {output_dir.absolute()}")
    print(f"Average: {elapsed / frame_index:.2f}s per frame")


if __name__ == "__main__":
    main()
