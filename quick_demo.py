"""Quick demo of PngDataIterator Python bindings."""
# cmk move or remove

import sys
import io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from busy_beaver_blaze import (
    PngDataIterator,
    BB5_CHAMP,
    log_step_iterator,
    create_frame,
    RESOLUTION_TINY,
)

# Create output directory
output_dir = Path("output") / "quick_demo"
output_dir.mkdir(parents=True, exist_ok=True)

# Generate a few frames
print("Generating 5 frames of BB5 Champion...")
frame_steps = log_step_iterator(10000, 5)
print(f"Frame steps: {frame_steps}")

iterator = PngDataIterator(
    early_stop=10000,
    program=BB5_CHAMP,
    width=RESOLUTION_TINY[0],
    height=RESOLUTION_TINY[1],
    pixel_policy="binning",
    frame_steps=frame_steps,
    colors=[],
    part_count=2  # Use 2 workers
)

for i, (step_idx, png_bytes) in enumerate(iterator):
    print(f"  Frame {i}: step {step_idx + 1:,}, {len(png_bytes):,} bytes")
    
    # Add text overlay
    frame = create_frame(png_bytes, "BB5", step_idx, RESOLUTION_TINY[0], RESOLUTION_TINY[1])
    
    # Save
    output_path = output_dir / f"frame_{i:03d}.png"
    frame.save(output_path)
    print(f"    Saved to {output_path}")

print(f"\n✅ Done! Check {output_dir.absolute()}")
