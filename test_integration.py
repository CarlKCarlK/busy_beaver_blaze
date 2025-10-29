"""Test the complete busy_beaver_blaze package."""

import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from busy_beaver_blaze import (
    # Pure Python
    Machine,
    Tape,
    # Rust bindings
    PngDataIterator,
    BB5_CHAMP,
    BB6_CONTENDER,
    # Frame utilities
    log_step_iterator,
    create_frame,
    blend_images,
    RESOLUTION_2K,
    RESOLUTION_4K,
)

print("✓ Successfully imported all exports")

# Test log_step_iterator
steps = log_step_iterator(1000, 10)
print(f"✓ log_step_iterator: {len(steps)} steps from 0 to {steps[-1]}")

# Test PngDataIterator
iterator = PngDataIterator(
    early_stop=100,
    program=BB5_CHAMP,
    width=RESOLUTION_2K[0],
    height=RESOLUTION_2K[1],
    pixel_policy="binning",
    frame_steps=[0, 10],
    colors=[],
    part_count=1
)
print("✓ Created PngDataIterator")

results = list(iterator)
step_idx, png_bytes = results[0]
print(f"✓ Got frame at step {step_idx}, size {len(png_bytes)} bytes")

# Test create_frame
frame = create_frame(png_bytes, "Test", step_idx, 320, 180)
print(f"✓ create_frame: {frame.size} {frame.mode}")

# Test blend_images
blended = blend_images(frame, frame, 0.5)
print(f"✓ blend_images: {blended.size} {blended.mode}")

# Test pure Python Machine with standard format
simple_program = "1RB1LC_1RC1RB_1RD0LE_1LA1LD_1RH0LA"  # BB5 standard format
machine = Machine(simple_program)
next(machine)
print(f"✓ Pure Python Machine: step completed")

print("\n✅ All integration tests passed!")
