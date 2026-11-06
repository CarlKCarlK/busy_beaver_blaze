"""Save raw sequential Square Root PNGs for reusable movie post-processing.

Run from the project root:
    uv run python example_python/movie_sqrt.py --steps 10000 --frames 20
"""

import argparse
import json
from pathlib import Path

from busy_beaver_blaze import (
    BB5_CHAMP,
    SqrtPngDataIterator,
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
    frame_steps = log_step_iterator(args.steps, args.frames)
    frames = SqrtPngDataIterator(
        frame_steps,
        program=args.program,
        resolution=resolution,
        pixel_policy=args.pixel_policy,
    )
    # A fresh directory prevents mixing frames from different runs.
    args.out.mkdir(parents=True, exist_ok=False)
    count = 0
    for frame_index, (step_index, png_bytes) in enumerate(frames):
        assert step_index == frame_steps[frame_index]
        # Save the exact encoder output before any filters, resizing or text.
        (args.out / f"frame_{frame_index:07d}.png").write_bytes(png_bytes)
        count += 1
        if count % 60 == 0 or count == args.frames:
            print(f"Saved {count}/{args.frames}: {step_index + 1:,} samples", flush=True)
    assert count == args.frames, "machine halted before all requested frames"
    manifest = {
        "format_version": 1, "raw": True, "time_scale": "sqrt",
        "program": args.program, "resolution": list(resolution),
        "pixel_policy": args.pixel_policy, "frame_steps": frame_steps,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
