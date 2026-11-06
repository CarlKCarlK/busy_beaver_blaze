"""Make a stretched 4K comparison with text drawn after sizing each diagram.

Reads saved raw Linear and Square Root PNGs and their manifests. This script
does not import a machine or frame iterator: all simulation happens separately.
All text is drawn at the final resolution, so only the diagrams stretch.
Enlargement keeps cell edges sharp. Both panels show the same step per frame.
"""

import argparse
import json
import subprocess
from pathlib import Path
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from busy_beaver_blaze import (
    add_linear_step_axis, add_sqrt_step_axis, linear_time_extent,
)

from busy_beaver_blaze.movie_encoding import (
    add_encoding_arguments, ordered_rgb_frames, video_encoder_args,
)


@lru_cache(maxsize=2)
def movie_font(size):
    font_path = Path(__file__).resolve().parents[1] / "examples/fonts/FiraCode-Regular.ttf"
    return ImageFont.truetype(str(font_path), size)


def render_comparison(task):
    linear_source, sqrt_source, source_index, step_index, resampling = task
    source = linear_source / f"frame_{source_index:07d}.png"
    with Image.open(source) as linear_raw:
        extent = linear_time_extent(step_index + 1, linear_raw.height)
        linear = add_linear_step_axis(
            linear_raw, step_index, time_extent=extent, resolution=(1920, 2160),
            resampling=resampling,
        )
    with Image.open(sqrt_source / f"frame_{source_index:07d}.png") as sqrt_raw:
        sqrt = add_sqrt_step_axis(
            sqrt_raw, step_index, resolution=(1920, 2160), resampling=resampling,
        )
    comparison = Image.new("RGB", (3840, 2160), "white")
    comparison.paste(linear, (0, 0))
    comparison.paste(sqrt, (1920, 0))
    draw = ImageDraw.Draw(comparison)
    for panel_index, heading in enumerate(["Linear", "Square Root"]):
        offset = panel_index * 1920
        draw.text(
            (offset + 960, 24), heading, font=movie_font(52),
            fill=(110, 110, 110), anchor="mt",
        )
        draw.text(
            (offset + 48, 2040), f"{step_index + 1:,}", font=movie_font(70),
            fill=(110, 110, 110), anchor="ls",
        )
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-source", type=Path, required=True)
    parser.add_argument("--sqrt-source", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1800)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold-seconds", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--resampling", choices=["cells", "adaptive"], default="cells")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--limit-frames", type=int)
    add_encoding_arguments(parser)
    args = parser.parse_args()
    if min(args.frames, args.fps) < 1 or args.hold_seconds < 0:
        parser.error("counts and fps must be positive; hold must be nonnegative")
    if min(args.workers, args.cpu_threads) < 1 or args.start_frame < 0:
        parser.error("workers and cpu-threads must be positive; start-frame must be nonnegative")
    if args.limit_frames is not None and args.limit_frames < 1:
        parser.error("limit-frames must be positive")
    if args.out.exists():
        parser.error("--out must be a new file")

    manifests = []
    for source, scale in [(args.linear_source, "linear"), (args.sqrt_source, "sqrt")]:
        manifest = json.loads((source / "manifest.json").read_text())
        assert manifest["format_version"] == 1 and manifest["raw"] is True
        assert manifest["time_scale"] == scale
        assert manifest["frame_steps"], "cache must contain frames"
        for source_index in range(len(manifest["frame_steps"])):
            assert (source / f"frame_{source_index:07d}.png").is_file(), "incomplete PNG cache"
        manifests.append(manifest)
    for key in ["program", "resolution", "pixel_policy", "frame_steps"]:
        assert manifests[0][key] == manifests[1][key], f"cache mismatch: {key}"
    source_steps = manifests[0]["frame_steps"]
    source_count = len(source_steps)
    source_indices = [
        source_count - 1 if args.frames == 1 else (
            frame_index * (source_count - 1) + (args.frames - 1) // 2
        ) // (args.frames - 1)
        for frame_index in range(args.frames)
    ]
    stop = None if args.limit_frames is None else args.start_frame + args.limit_frames
    source_indices = source_indices[args.start_frame:stop]
    assert source_indices, "selected frame range is empty"
    steps = [source_steps[index] for index in source_indices]
    frame_count = len(steps)
    tasks = [
        (args.linear_source, args.sqrt_source, source_index, step_index, args.resampling)
        for source_index, step_index in zip(source_indices, steps, strict=True)
    ]
    encoder_name, encoder_options = video_encoder_args(args.encoder, args.cpu_threads)
    print(f"Preparing frames with {args.workers} processes; encoder: {encoder_name}", flush=True)
    hold_frames = args.hold_seconds * args.fps
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-n",
        "-filter_threads", str(min(8, args.cpu_threads)), "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "3840x2160",
        "-framerate", str(args.fps), "-i", "pipe:0",
        "-frames:v", str(frame_count + hold_frames),
        *encoder_options,
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        assert encoder.stdin is not None
        frames = ordered_rgb_frames(tasks, render_comparison, args.workers, 3840 * 2160 * 3)
        for frame_index, frame_bytes in enumerate(frames):
            encoder.stdin.write(frame_bytes)
            if frame_index + 1 == frame_count:
                final_frame = bytes(frame_bytes)
            if (frame_index + 1) % 60 == 0 or frame_index + 1 == frame_count:
                print(f"Encoded {frame_index + 1}/{frame_count}: {steps[frame_index] + 1:,} samples", flush=True)
        for _ in range(hold_frames):
            encoder.stdin.write(final_frame)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise SystemExit("FFmpeg encoding failed")
    if args.preview is not None:
        Image.frombytes("RGB", (3840, 2160), final_frame).save(args.preview)


if __name__ == "__main__":
    main()
