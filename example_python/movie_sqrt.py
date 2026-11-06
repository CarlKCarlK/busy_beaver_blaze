"""Save sequential Square Root frames for video encoding.

Run from the project root:
    uv run python example_python/movie_sqrt.py --steps 10000 --frames 20
"""

import argparse
from pathlib import Path

from busy_beaver_blaze import (
    BB5_CHAMP,
    SqrtPngDataIterator,
    add_sqrt_step_axis,
    create_frame,
    log_step_iterator,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=47_176_870)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--program", default=BB5_CHAMP)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--pixel-policy", choices=["binning", "sampling"], default="binning")
    parser.add_argument("--out", type=Path, default=Path("output/movie_sqrt"))
    args = parser.parse_args()
    if args.steps < 1 or args.frames < 1:
        parser.error("--steps and --frames must be positive")

    resolution = (args.width, args.height)
    frames = SqrtPngDataIterator(
        log_step_iterator(args.steps, args.frames),
        program=args.program,
        resolution=resolution,
        pixel_policy=args.pixel_policy,
    )
    # A fresh directory prevents mixing frames from different runs.
    args.out.mkdir(parents=True, exist_ok=False)
    for frame_index, (step_index, png_bytes) in enumerate(frames):
        frame = add_sqrt_step_axis(
            create_frame(png_bytes, "", step_index, resolution), step_index,
        )
        frame.save(args.out / f"frame_{frame_index:07d}.png")
        print(f"Frame {frame_index}: {step_index + 1:,} samples")


if __name__ == "__main__":
    main()
