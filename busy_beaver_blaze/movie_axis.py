"""Post-processing for Square Root movie frames."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _sqrt_step_ticks(step_count: int, plot_height: int) -> list[tuple[int, int]]:
    """Return integer sample boundaries and their inverse-mapped pixel positions."""
    assert step_count > 0
    assert plot_height > 1
    # Keep equal machine-step intervals through each decade: 1..10, 10..100,
    # etc. Their screen positions are deliberately uneven under Square Root.
    interval = 10 ** max(0, len(str(step_count - 1)) - 1)
    steps = range(0, step_count + 1, interval)
    return [
        (step, round(math.sqrt(step / step_count) * (plot_height - 1)))
        for step in steps
    ]


def _step_label(step: int) -> str:
    """Compact decade ticks without losing precision."""
    for unit, suffix in [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]:
        if step >= unit and step % unit == 0:
            return f"{step // unit}{suffix}"
    return str(step)


def add_sqrt_step_axis(frame: Image.Image, step_index: int) -> Image.Image:
    """Add a left-side Machine steps axis, preserving the frame's dimensions.

    The input must show a complete Square Root diagram, with no existing axis.
    ``step_index`` is the zero-based index yielded by SqrtPngDataIterator.
    As in the renderer and create_frame caption, the timeline includes the
    initial blank sample: its domain is [0, step_index + 1]. Output ticks mark
    boundaries at time = (step_index + 1) * vertical_fraction**2.
    Ticks use equal step intervals, increasing by a factor of ten each decade;
    e.g. 10M, 20M, ..., 100M have uneven vertical spacing. The caption retains
    the exact current count when it falls between ticks.

    The diagram is resized into a fixed inset so it does not move as labels
    acquire more digits. This is purely image post-processing; no history or
    simulation is needed. The input image is not modified.

    Raises:
        ValueError: If the index is negative or the frame is smaller than 160x90.
    """
    if step_index < 0:
        raise ValueError("step_index must be nonnegative")
    width, height = frame.size
    if width < 160 or height < 90:
        raise ValueError("Axis frames must be at least 160x90")

    font_size = max(10, round(24 * height / 1080))
    font_path = Path(__file__).resolve().parent.parent / "examples/fonts/FiraCode-Regular.ttf"
    font = (
        ImageFont.truetype(str(font_path), font_size)
        if font_path.is_file()
        else ImageFont.load_default(size=font_size)
    )
    padding = max(6, round(12 * height / 1080))
    margin = max(round(width * 0.14), font_size * 8)
    plot_top = font_size * 3 + padding
    plot_bottom = height - padding - font_size
    plot_height = plot_bottom - plot_top + 1
    plot_width = width - margin - padding
    assert plot_height > 1 and plot_width > 0

    output = Image.new("RGB", frame.size, "white")
    plot = frame.convert("RGB").resize((plot_width, plot_height), Image.Resampling.LANCZOS)
    output.paste(plot, (margin, plot_top))
    draw = ImageDraw.Draw(output)
    ink = (65, 65, 65)
    axis_x = margin - padding
    draw.multiline_text(
        (padding, padding), "Machine\nsteps", font=font, fill=ink, spacing=2,
    )
    draw.line((axis_x, plot_top, axis_x, plot_bottom), fill=ink)
    previous_label_y = -height
    ticks = _sqrt_step_ticks(step_index + 1, plot_height)
    has_bottom_tick = ticks[-1][0] == step_index + 1
    for step, pixel_y in ticks:
        label_y = plot_top + pixel_y
        tick_length = max(4, padding // 2)
        draw.line((axis_x - tick_length, label_y, axis_x, label_y), fill=ink)
        # Preserve the endpoints while avoiding overlapping labels in small frames.
        if step not in (0, step_index + 1) and (
            label_y - previous_label_y < font_size * 2
            or (has_bottom_tick and plot_bottom - label_y < font_size * 2)
        ):
            continue
        label = _step_label(step)
        label_font = font
        available_width = axis_x - tick_length - 2 * padding
        if draw.textlength(label, font=label_font) > available_width:
            label_font = ImageFont.load_default(size=max(
                6, int(font_size * available_width / draw.textlength(label, font=font)),
            ))
        draw.text(
            (axis_x - tick_length - padding, label_y), label,
            font=label_font, fill=ink, anchor="rm",
        )
        previous_label_y = label_y
    return output
