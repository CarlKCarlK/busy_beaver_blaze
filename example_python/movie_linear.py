"""Save raw Linear movie frames using the parallel Rust iterator.

These retain native dimensions, including the partial final time row. Encode
with post_sqrt_movie.py --time-scale linear to resize, caption, add the axis,
and append the final hold without rerunning the machine.
"""

import argparse
import json
from pathlib import Path

from busy_beaver_blaze import PngDataIterator, log_step_iterator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps, args.frames, args.workers, args.height) < 1 or args.width < 2:
        parser.error("counts must be positive and width must be at least 2")
    args.out.mkdir(parents=True, exist_ok=False)
    frame_steps = log_step_iterator(args.steps, args.frames)
    frames = PngDataIterator(
        frame_steps, program=args.program,
        resolution=(args.width, args.height), early_stop=args.steps,
        pixel_policy="binning", part_count=args.workers,
    )
    count = 0
    for frame_index, (step_index, png) in enumerate(frames):
        assert step_index == frame_steps[frame_index]
        (args.out / f"frame_{frame_index:07d}.png").write_bytes(png)
        count += 1
        if count % 60 == 0 or count == args.frames:
            print(f"Saved {count}/{args.frames}: {step_index + 1:,} samples", flush=True)
    assert count == args.frames, "machine halted before all requested frames"
    manifest = {
        "format_version": 1, "raw": True, "time_scale": "linear",
        "program": args.program, "resolution": [args.width, args.height],
        "pixel_policy": "binning", "frame_steps": frame_steps,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
