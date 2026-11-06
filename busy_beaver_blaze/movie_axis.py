"""Post-processing for Square Root movie frames."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _step_interval(step_count: int) -> int:
    assert step_count > 0
    return 10 ** max(0, len(str(step_count - 1)) - 1)


def _sqrt_step_ticks(step_count: int, plot_height: int) -> list[tuple[int, int]]:
    """Return integer sample boundaries and their inverse-mapped pixel positions."""
    assert step_count > 0
    assert plot_height > 1
    # Keep equal machine-step intervals through each decade: 1..10, 10..100,
    # etc. Their screen positions are deliberately uneven under Square Root.
    interval = _step_interval(step_count)
    steps = range(0, step_count + 1, interval)
    return [
        (step, round(math.sqrt(step / step_count) * (plot_height - 1)))
        for step in steps
    ]


def _step_label(step: int) -> str:
    """Write the full machine-step count with comma separators."""
    return f"{step:,}"


def _fading_ticks(step_count: int, plot_height: int) -> tuple[list[tuple[int, int]], float]:
    """Fade the previous grid over the first tenth of the next log decade."""
    interval = _step_interval(step_count)
    if interval == 1:
        return [], 0.0
    opacity = max(0.0, 1.0 - math.log10(step_count / interval) / 0.1)
    if opacity == 0.0:
        return [], 0.0
    # Shared zero and decade-boundary ticks belong to the new scale. Only the
    # nine intermediate ticks disappear; do not grow the old grid indefinitely.
    ticks = [
        (step, round(math.sqrt(step / step_count) * (plot_height - 1)))
        for step in range(interval // 10, interval, interval // 10)
    ]
    return ticks, opacity


def add_sqrt_step_axis(frame: Image.Image, step_index: int) -> Image.Image:
    """Add a right-side Machine steps axis, preserving the frame's dimensions.

    The input must show a complete Square Root diagram, with no existing axis.
    ``step_index`` is the zero-based index yielded by SqrtPngDataIterator.
    As in the renderer and create_frame caption, the timeline includes the
    initial blank sample: its domain is [0, step_index + 1]. Output ticks mark
    boundaries at time = (step_index + 1) * vertical_fraction**2.
    Ticks use equal step intervals, increasing by a factor of ten each decade;
    e.g. 10,000,000, 20,000,000, ..., 100,000,000 have uneven vertical spacing.
    Old intermediate ticks fade to white over the first tenth of each new log
    decade (up to about 1.26 times the boundary), while new ticks stay solid.
    This deterministic fade works on saved frames without previous-frame state.
    The caption retains the exact current count when it falls between ticks.

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
    output.paste(plot, (padding, plot_top))
    draw = ImageDraw.Draw(output)
    ink = (65, 65, 65)
    axis_x = padding + plot_width + padding
    draw.multiline_text(
        (axis_x + padding, padding), "Machine\nsteps", font=font, fill=ink, spacing=2,
    )
    draw.line((axis_x, plot_top, axis_x, plot_bottom), fill=ink)
    ticks = _sqrt_step_ticks(step_index + 1, plot_height)
    fading_ticks, opacity = _fading_ticks(step_index + 1, plot_height)
    fade_red, fade_green, fade_blue = (
        round(255 + (channel - 255) * opacity) for channel in ink
    )
    fade_ink = (fade_red, fade_green, fade_blue)
    for layer_ticks, layer_ink in [(fading_ticks, fade_ink), (ticks, ink)]:
        previous_label_y = -height
        has_bottom_tick = bool(layer_ticks) and layer_ticks[-1][0] == step_index + 1
        for step, pixel_y in layer_ticks:
            label_y = plot_top + pixel_y
            tick_length = max(4, padding // 2)
            draw.line((axis_x, label_y, axis_x + tick_length, label_y), fill=layer_ink)
            # Preserve endpoints while avoiding overlapping labels in small frames.
            if step not in (0, step_index + 1) and (
                label_y - previous_label_y < font_size * 2
                or (has_bottom_tick and plot_bottom - label_y < font_size * 2)
                or (layer_ticks is fading_ticks and any(
                    abs(pixel_y - current_y) < font_size * 2 for _, current_y in ticks
                ))
            ):
                continue
            label = _step_label(step)
            label_font = font
            label_x = axis_x + tick_length + padding
            available_width = width - label_x - padding
            if draw.textlength(label, font=label_font) > available_width:
                label_font = ImageFont.load_default(size=max(
                    6, int(font_size * available_width / draw.textlength(label, font=font)),
                ))
            draw.text(
                (label_x, label_y), label,
                font=label_font, fill=layer_ink, anchor="lm",
            )
            previous_label_y = label_y
    return output
