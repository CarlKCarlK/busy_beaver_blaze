"""Add a step axis to saved movie frames and stream them into an MP4.

Inputs are saved PNGs with logarithmic frame spacing. Raw PNG caches from
movie_sqrt.py and movie_linear.py include manifest.json. Older, already resized
Square Root PNGs are also supported. No Turing machine is rerun. FFmpeg must
be installed. Cell enlargement uses nearest-neighbor, with text drawn afterward.
"""

import argparse
import json
import subprocess
from pathlib import Path
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from busy_beaver_blaze import (
    add_linear_step_axis, add_sqrt_step_axis, linear_time_extent, log_step_iterator,
)

from busy_beaver_blaze.movie_encoding import (
    add_encoding_arguments, ordered_rgb_frames, video_encoder_args,
)


@lru_cache(maxsize=4)
def counter_font(size):
    font_path = Path(__file__).resolve().parents[1] / "examples/fonts/FiraCode-Regular.ttf"
    return ImageFont.truetype(str(font_path), size)


def render_frame(task):
    source, step_index, resolution, time_scale, raw_cache, resampling = task
    width, height = resolution
    with Image.open(source) as frame:
        if time_scale == "linear":
            time_extent = linear_time_extent(step_index + 1, frame.height)
            annotated = add_linear_step_axis(
                frame, step_index, time_extent=time_extent, resolution=resolution,
                resampling=resampling,
            )
        else:
            if not raw_cache:
                assert frame.size == resolution, "source frames must have identical dimensions"
            annotated = add_sqrt_step_axis(frame, step_index, resolution=resolution, resampling=resampling)
    if raw_cache:
        ImageDraw.Draw(annotated).text(
            (round(width * 0.025), round(height * 0.945)),
            f"{step_index + 1:,}", font=counter_font(max(10, round(50 * height / 1080))),
            fill=(110, 110, 110), anchor="ls",
        )
    return annotated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-count", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold-seconds", type=int, default=3)
    parser.add_argument("--time-scale", choices=["sqrt", "linear"], default="sqrt")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resampling", choices=["cells", "adaptive"], default="cells")
    add_encoding_arguments(parser)
    args = parser.parse_args()
    if min(args.source_count, args.steps, args.frames, args.fps) < 1:
        parser.error("counts, steps, and fps must be positive")
    if args.hold_seconds < 0:
        parser.error("--hold-seconds must be nonnegative")
    if min(args.workers, args.cpu_threads) < 1:
        parser.error("workers and cpu-threads must be positive")
    if args.out.exists():
        parser.error("--out must be a new file")

    steps = log_step_iterator(args.steps, args.source_count)
    manifest_path = args.source / "manifest.json"
    raw_cache = args.time_scale == "linear"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["format_version"] == 1 and manifest["raw"] is True
        assert manifest["time_scale"] == args.time_scale
        assert manifest["frame_steps"] == steps, "step schedule does not match saved cache"
        raw_cache = True
    if raw_cache:
        resolution = (args.width, args.height)
    else:
        with Image.open(args.source / "frame_0000000.png") as first:
            resolution = first.size
    width, height = resolution
    if width % 2 or height % 2:
        parser.error("H.264 output requires even image dimensions")
    tasks = []
    for frame_index in range(args.frames):
        source_index = args.source_count - 1 if args.frames == 1 else (
            frame_index * (args.source_count - 1) + (args.frames - 1) // 2
        ) // (args.frames - 1)
        tasks.append((
            args.source / f"frame_{source_index:07d}.png", steps[source_index], resolution,
            args.time_scale, raw_cache, args.resampling,
        ))
    encoder_name, encoder_options = video_encoder_args(args.encoder, args.cpu_threads)
    print(f"Preparing frames with {args.workers} processes; encoder: {encoder_name}", flush=True)

    hold_frames = args.hold_seconds * args.fps
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-n",
        "-filter_threads", str(min(8, args.cpu_threads)), "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(args.fps),
        "-i", "pipe:0", "-frames:v", str(args.frames + hold_frames),
        *encoder_options,
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        assert encoder.stdin is not None
        frames = ordered_rgb_frames(tasks, render_frame, args.workers, width * height * 3)
        for frame_index, frame_bytes in enumerate(frames):
            encoder.stdin.write(frame_bytes)
            if frame_index + 1 == args.frames:
                final_frame = bytes(frame_bytes)
            if (frame_index + 1) % 60 == 0 or frame_index + 1 == args.frames:
                print(f"Encoded {frame_index + 1}/{args.frames} frames", flush=True)
        for _ in range(hold_frames):
            encoder.stdin.write(final_frame)
        if hold_frames:
            print(f"Added {args.hold_seconds}-second final hold ({hold_frames} frames)", flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise SystemExit("FFmpeg encoding failed")


if __name__ == "__main__":
    main()
