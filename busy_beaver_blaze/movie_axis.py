"""Post-processing for Linear and Square Root movie frames."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _step_interval(step_count: int) -> int:
    assert step_count > 0
    return 10 ** max(0, len(str(step_count - 1)) - 1)


def _sqrt_step_ticks(step_count: int, plot_height: int) -> list[tuple[int, int]]:
    """Return integer sample boundaries and their inverse-mapped pixel positions."""
    return _step_ticks(step_count, plot_height, "sqrt", step_count)


def _step_pixel(step: int, time_extent: int, plot_height: int, time_scale: str) -> int:
    fraction = step / time_extent
    if time_scale == "sqrt":
        fraction = math.sqrt(fraction)
    else:
        assert time_scale == "linear"
    return round(fraction * (plot_height - 1))


def _step_ticks(
    step_count: int, plot_height: int, time_scale: str, time_extent: int,
) -> list[tuple[int, int]]:
    assert step_count > 0
    assert time_extent >= step_count
    assert plot_height > 1
    # Keep equal machine-step intervals through each decade: 1..10, 10..100,
    # etc. Their screen positions are deliberately uneven under Square Root.
    interval = _step_interval(step_count)
    steps = range(0, step_count + 1, interval)
    return [
        (step, _step_pixel(step, time_extent, plot_height, time_scale))
        for step in steps
    ]


def _step_label(step: int) -> str:
    """Write the full machine-step count with comma separators."""
    return f"{step:,}"


def _resize_diagram(
    frame: Image.Image, resolution: tuple[int, int], resampling: str = "cells",
) -> Image.Image:
    """Keep enlarged cells sharp; use area averaging only on shrinking axes."""
    width, height = resolution
    if resampling not in ("cells", "adaptive"):
        raise ValueError("resampling must be 'cells' or 'adaptive'")
    frame = frame.convert("RGB")
    if width != frame.width:
        method = Image.Resampling.NEAREST if width > frame.width else Image.Resampling.BOX
        if resampling == "adaptive" and frame.width < width < 4 * frame.width:
            method = Image.Resampling.LANCZOS
        frame = frame.resize((width, frame.height), method)
    if height != frame.height:
        method = Image.Resampling.NEAREST if height > frame.height else Image.Resampling.BOX
        if resampling == "adaptive" and frame.height < height < 4 * frame.height:
            method = Image.Resampling.LANCZOS
        frame = frame.resize((frame.width, height), method)
    return frame


def _fading_ticks(
    step_count: int, plot_height: int, time_scale: str = "sqrt", time_extent: int | None = None,
) -> tuple[list[tuple[int, int]], float]:
    """Fade the previous grid over the first tenth of the next log decade."""
    interval = _step_interval(step_count)
    if time_extent is None:
        time_extent = step_count
    if interval == 1:
        return [], 0.0
    opacity = max(0.0, 1.0 - math.log10(step_count / interval) / 0.1)
    if opacity == 0.0:
        return [], 0.0
    # Shared zero and decade-boundary ticks belong to the new scale. Only the
    # nine intermediate ticks disappear; do not grow the old grid indefinitely.
    ticks = [
        (step, _step_pixel(step, time_extent, plot_height, time_scale))
        for step in range(interval // 10, interval, interval // 10)
    ]
    return ticks, opacity


def add_sqrt_step_axis(
    frame: Image.Image, step_index: int, *, resolution: tuple[int, int] | None = None,
    resampling: str = "cells",
) -> Image.Image:
    """Add a right-side Machine steps axis, preserving the frame's dimensions.

    The input must show a complete Square Root diagram, with no existing axis.
    ``step_index`` is the zero-based index yielded by SqrtPngDataIterator.
    As in the renderer and create_frame caption, the timeline includes the
    initial blank sample: its domain is [0, step_index + 1]. Output ticks mark
    boundaries at time = (step_index + 1) * vertical_fraction**2.
    Ticks use equal step intervals, increasing by a factor of ten each decade;
    e.g. 10,000,000, 20,000,000, ..., 100,000,000 have uneven vertical spacing.
    Old intermediate ticks fade out over the first tenth of each new log
    decade (up to about 1.26 times the boundary), while new ticks stay solid.
    This deterministic fade works on saved frames without previous-frame state.
    The caption retains the exact current count when it falls between ticks.
    Labels sit to the left of the right-side scale, over the diagram, in muted
    gray. Fading uses transparency so the diagram remains visible underneath.

    The diagram is resized into a fixed inset so it does not move as labels
    acquire more digits. This is purely image post-processing; no history or
    simulation is needed. The input image is not modified.
    ``resolution`` optionally sets the final canvas size: only the diagram is
    resized, and text is drawn afterward with undistorted glyphs.
    ``resampling="cells"`` keeps enlarged cells sharp. ``"adaptive"`` interpolates
    modest enlargements of dense data, retaining nearest-neighbor above 4x.

    Raises:
        ValueError: If the index is negative or the frame is smaller than 160x90.
    """
    return _add_step_axis(frame, step_index, "sqrt", step_index + 1, resolution, resampling)


def linear_time_extent(step_count: int, source_height: int) -> int:
    """Recover the padded Linear time extent from a raw native PNG's height.

    Native rows cover equal power-of-two step intervals; only the last row can
    be incomplete. Call this before resizing the PNG to the movie dimensions.
    """
    assert step_count > 0 and source_height > 0
    minimum_stride = (step_count + source_height - 1) // source_height
    stride = 2 ** (minimum_stride - 1).bit_length()
    assert (source_height - 1) * stride < step_count <= source_height * stride
    return source_height * stride


def add_linear_step_axis(
    frame: Image.Image, step_index: int, *, time_extent: int | None = None,
    resolution: tuple[int, int] | None = None,
    resampling: str = "cells",
) -> Image.Image:
    """Add the same fading, overlaid labels to a Linear time diagram.

    ``time_extent`` includes the native PNG's partially filled final row. Use
    ``linear_time_extent(step_index + 1, raw_png.height)`` before resizing.
    If omitted, the frame is assumed to cover exactly ``step_index + 1`` samples.
    ``resolution`` resizes the diagram before drawing the axis and its text.
    """
    if time_extent is None:
        time_extent = step_index + 1
    if time_extent < step_index + 1:
        raise ValueError("time_extent must cover every sample")
    return _add_step_axis(frame, step_index, "linear", time_extent, resolution, resampling)


def _add_step_axis(
    frame: Image.Image, step_index: int, time_scale: str, time_extent: int,
    resolution: tuple[int, int] | None = None,
    resampling: str = "cells",
) -> Image.Image:
    if step_index < 0:
        raise ValueError("step_index must be nonnegative")
    width, height = frame.size if resolution is None else resolution
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
    plot_top = font_size * 2 + padding
    plot_bottom = height - padding - font_size
    plot_height = plot_bottom - plot_top + 1
    plot_width = width - 3 * padding
    assert plot_height > 1 and plot_width > 0

    output = Image.new("RGB", (width, height), "white")
    plot = _resize_diagram(frame, (plot_width, plot_height), resampling)
    output.paste(plot, (padding, plot_top))
    draw = ImageDraw.Draw(output)
    ink = (110, 110, 110)
    axis_x = padding + plot_width + padding
    draw.text(
        (axis_x, padding), "Machine steps", font=font, fill=ink, anchor="rt",
    )
    draw.line((axis_x, plot_top, axis_x, plot_bottom), fill=ink)
    ticks = _step_ticks(step_index + 1, plot_height, time_scale, time_extent)
    fading_ticks, opacity = _fading_ticks(step_index + 1, plot_height, time_scale, time_extent)
    labels = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(labels)
    for layer_ticks, alpha in [(fading_ticks, round(255 * opacity)), (ticks, 255)]:
        layer_ink = (*ink, alpha)
        previous_label_y = -height
        has_bottom_tick = bool(layer_ticks) and layer_ticks[-1][0] == step_index + 1
        for step, pixel_y in layer_ticks:
            label_y = plot_top + pixel_y
            tick_length = max(4, padding // 2)
            draw.line((axis_x - tick_length, label_y, axis_x, label_y), fill=layer_ink)
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
            label_x = axis_x - tick_length - padding
            available_width = label_x - padding
            if draw.textlength(label, font=label_font) > available_width:
                label_font = ImageFont.load_default(size=max(
                    6, int(font_size * available_width / draw.textlength(label, font=font)),
                ))
            anchor = "rb" if step == 0 else "rt" if step == step_index + 1 else "rm"
            draw.text(
                (label_x, label_y), label,
                font=label_font, fill=layer_ink, anchor=anchor,
            )
            previous_label_y = label_y
    return Image.alpha_composite(output.convert("RGBA"), labels).convert("RGB")
