"""Add a step axis to saved Square Root frames and stream them into an MP4.

The source PNGs must be unannotated frames from movie_sqrt.py before the axis
was added, with logarithmic frame spacing. No Turing machine is rerun and no
additional PNG sequence is written. FFmpeg must be installed.
"""

import argparse
import subprocess
from pathlib import Path

from PIL import Image

from busy_beaver_blaze import add_sqrt_step_axis, log_step_iterator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-count", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.source_count, args.steps, args.frames, args.fps) < 1:
        parser.error("counts, steps, and fps must be positive")
    if args.out.exists():
        parser.error("--out must be a new file")

    steps = log_step_iterator(args.steps, args.source_count)
    with Image.open(args.source / "frame_0000000.png") as first:
        resolution = first.size
    width, height = resolution
    if width % 2 or height % 2:
        parser.error("H.264 output requires even image dimensions")

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(args.fps),
        "-i", "pipe:0", "-frames:v", str(args.frames),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        assert encoder.stdin is not None
        for frame_index in range(args.frames):
            if args.frames == 1:
                source_index = args.source_count - 1
            else:
                # Include both endpoints when changing the movie duration.
                source_index = (
                    frame_index * (args.source_count - 1) + (args.frames - 1) // 2
                ) // (args.frames - 1)
            with Image.open(args.source / f"frame_{source_index:07d}.png") as frame:
                assert frame.size == resolution, "source frames must have identical dimensions"
                annotated = add_sqrt_step_axis(frame, steps[source_index])
            encoder.stdin.write(annotated.tobytes())
            if (frame_index + 1) % 60 == 0 or frame_index + 1 == args.frames:
                print(f"Encoded {frame_index + 1}/{args.frames} frames", flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise SystemExit("FFmpeg encoding failed")


if __name__ == "__main__":
    main()
