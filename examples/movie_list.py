# cmk is this the right location for this example?
"""Generate Turing machine visualization frames using Python + Rust.

This script demonstrates using the PyO3 bindings to generate high-quality
visualization frames. It's a simplified Python version of examples/movie_list.rs
that focuses on a single Turing machine rather than batch processing.

Example usage:
    python examples/movie_list.py
"""

import time
from pathlib import Path

from busy_beaver_blaze import (
    PngDataIterator,
    BB5_CHAMP,
    BB6_CONTENDER,
    log_step_iterator,
    create_frame,
    blend_images,
    RESOLUTION_2K,
    RESOLUTION_4K,
)


def main():
    """Generate visualization frames for a Turing machine."""
    start_time = time.time()
    
    # Configuration
    output_dir = Path("output") / "bb5_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    program = BB5_CHAMP
    width, height = RESOLUTION_2K  # Use 2K resolution
    early_stop = 47_176_870  # BB5 halts at this step
    frame_count = 100
    pixel_policy = "binning"  # "binning" or "sampling"
    colors = []  # Empty list uses default colors
    part_count = 8  # Number of parallel workers (0 = auto-detect CPU count)
    caption = "BB5 Champion"
    
    # Generate logarithmically-spaced step indices
    print(f"Generating {frame_count} frames up to step {early_stop:,}")
    frame_steps = log_step_iterator(early_stop, frame_count)
    print(f"Frame steps: {frame_steps[:5]}...{frame_steps[-5:]}")
    
    # Create iterator
    print(f"Creating PngDataIterator with {part_count} workers...")
    iterator = PngDataIterator(
        early_stop=early_stop,
        program=program,
        width=width,
        height=height,
        pixel_policy=pixel_policy,
        frame_steps=frame_steps,
        colors=colors,
        part_count=part_count,
    )
    
    # Process frames
    print(f"Processing frames and saving to {output_dir}/")
    last_frame = None
    frame_index = 0
    
    for step_index, png_bytes in iterator:
        elapsed = time.time() - start_time
        print(f"Frame {frame_index:04d}, Step {step_index + 1:,}, "
              f"{len(png_bytes):,} bytes, {elapsed:.1f}s elapsed")
        
        # Create frame with text overlay
        frame = create_frame(
            png_bytes,
            caption,
            step_index,
            width,
            height
        )
        
        # Optional: Create transition frames from previous frame
        if last_frame is not None and frame_index > 0:
            TRANSITION_FRAMES = 5
            for i in range(TRANSITION_FRAMES):
                fraction = (i + 1) / (TRANSITION_FRAMES + 1)
                blended = blend_images(last_frame, frame, fraction)
                transition_path = output_dir / f"frame_{frame_index:04d}_transition_{i:02d}.png"
                blended.save(transition_path)
        
        # Save main frame
        frame_path = output_dir / f"frame_{frame_index:04d}.png"
        frame.save(frame_path)
        
        last_frame = frame
        frame_index += 1
    
    elapsed = time.time() - start_time
    print(f"\nCompleted {frame_index} frames in {elapsed:.1f}s")
    print(f"Output saved to: {output_dir.absolute()}")
    print(f"Average: {elapsed / frame_index:.2f}s per frame")


if __name__ == "__main__":
    main()
