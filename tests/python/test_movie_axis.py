"""The movie axis follows the renderer's nonlinear sample boundaries."""

import pytest
from PIL import Image

from busy_beaver_blaze import add_linear_step_axis, add_sqrt_step_axis, linear_time_extent
from busy_beaver_blaze.movie_axis import (
    _fading_ticks, _resize_diagram, _sqrt_step_ticks, _step_label, _step_ticks,
)


def test_square_root_mapping():
    assert _sqrt_step_ticks(100_000_000, 1001) == [
        (0, 0), (10_000_000, 316), (20_000_000, 447), (30_000_000, 548),
        (40_000_000, 632), (50_000_000, 707), (60_000_000, 775),
        (70_000_000, 837), (80_000_000, 894), (90_000_000, 949),
        (100_000_000, 1000),
    ]


def test_linear_ticks_are_equally_spaced():
    assert _step_ticks(100_000_000, 1001, "linear", 100_000_000) == [
        (step * 10_000_000, step * 100) for step in range(11)
    ]
    # The last of five 8-step rows is only partly filled at sample 33.
    assert linear_time_extent(33, 5) == 40
    assert _step_ticks(33, 1001, "linear", 40) == [
        (0, 0), (10, 250), (20, 500), (30, 750),
    ]


@pytest.mark.parametrize("samples, source_height, expected", [
    (1, 1, 1), (7, 7, 7), (31, 4, 32), (32, 4, 32), (33, 5, 40),
    (100_000_000, 1526, 100_007_936),
])
def test_linear_native_row_extent(samples, source_height, expected):
    assert linear_time_extent(samples, source_height) == expected


@pytest.mark.parametrize("workers", [1, 2])
def test_linear_axis_with_real_native_frames(workers):
    from io import BytesIO
    from busy_beaver_blaze import PngDataIterator

    frames = PngDataIterator(
        [0, 30, 31, 32], program="1RA1RA", early_stop=34,
        resolution=(16, 4), part_count=workers,
    )
    for step_index, png in frames:
        with Image.open(BytesIO(png)) as source:
            extent = linear_time_extent(step_index + 1, source.height)
        assert extent == {0: 1, 30: 32, 31: 32, 32: 40}[step_index]
        annotated = add_linear_step_axis(Image.new("RGB", (320, 180)), step_index, time_extent=extent)
        assert annotated.size == (320, 180)


@pytest.mark.parametrize("step_count", [1, 2, 3, 4, 17, 1_000_000])
def test_early_frames_and_endpoints(step_count):
    ticks = _sqrt_step_ticks(step_count, 1001)
    assert ticks[0] == (0, 0)
    assert ticks[-1][0] <= step_count
    if step_count != 17:
        assert ticks[-1] == (step_count, 1000)
    assert len({step for step, _ in ticks}) == len(ticks)
    for step, pixel_y in ticks:
        # Check the forward mapping independently, allowing half a pixel of rounding.
        lower = step_count * (max(0, pixel_y - 0.5) / 1000) ** 2
        upper = step_count * (min(1000, pixel_y + 0.5) / 1000) ** 2
        assert lower <= step <= upper


@pytest.mark.parametrize("step_count, interval", [
    (10, 1), (11, 10), (100, 10), (101, 100),
    (1_000_000, 100_000), (100_000_000, 10_000_000),
])
def test_tick_intervals_change_at_decades(step_count, interval):
    ticks = _sqrt_step_ticks(step_count, 1001)
    assert [step for step, _ in ticks] == list(range(0, step_count + 1, interval))


def test_million_labels():
    ticks = _sqrt_step_ticks(100_000_000, 1001)
    assert [_step_label(step) for step, _ in ticks] == [
        "0", "10,000,000", "20,000,000", "30,000,000", "40,000,000", "50,000,000",
        "60,000,000", "70,000,000", "80,000,000", "90,000,000", "100,000,000",
    ]


@pytest.mark.parametrize("size", [(320, 180), (1920, 1080)])
def test_axis_preserves_resolution_input_and_plot_content(size):
    frame = Image.new("RGB", size, "orange")
    before = frame.tobytes()
    annotated = add_sqrt_step_axis(frame, 99_999_999)
    assert annotated.size == size
    assert frame.tobytes() == before
    assert annotated.getpixel((size[0] // 2, size[1] // 2)) == (255, 165, 0)
    assert annotated.getpixel((0, 0)) == (255, 255, 255)
    assert annotated.crop((size[0] * 7 // 8, 0, size[0], size[1])).getextrema()[0][0] < 150
    assert annotated.getpixel((0, size[1] // 2)) == (255, 255, 255)
    assert annotated.getpixel((size[0] * 9 // 10, size[1] // 2)) == (255, 165, 0)


@pytest.mark.parametrize("boundary", [10, 100, 1_000_000, 10_000_000])
def test_old_grid_fades_as_new_decade_begins(boundary):
    early_ticks, early_opacity = _fading_ticks(boundary + 1, 1001)
    later_ticks, later_opacity = _fading_ticks(boundary * 12 // 10, 1001)
    assert [step for step, _ in early_ticks] == list(range(boundary // 10, boundary, boundary // 10))
    assert 0 < later_opacity < early_opacity < 1
    assert len(later_ticks) == 9
    assert _fading_ticks(boundary * 13 // 10, 1001) == ([], 0.0)
    assert not set(step for step, _ in early_ticks) & set(
        step for step, _ in _sqrt_step_ticks(boundary + 1, 1001)
    )


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="nonnegative"):
        add_sqrt_step_axis(Image.new("RGB", (320, 180)), -1)
    with pytest.raises(ValueError, match="160x90"):
        add_sqrt_step_axis(Image.new("RGB", (16, 8)), 0)


def test_fading_labels_blend_with_tape_color():
    frame = Image.new("RGB", (1920, 1080), "orange")
    annotated = add_sqrt_step_axis(frame, 109)
    # The old 10-step label is around this location at a 110-sample frame.
    # Its solid glyph pixels should be muted orange/gray, not an opaque gray
    # or white mask painted over the tape.
    crop = annotated.crop((1750, 340, 1895, 375))
    colors = crop.getcolors(crop.width * crop.height)
    assert colors is not None
    assert any(
        count > 5 and 160 <= red <= 180 and 125 <= green <= 140 and 55 <= blue <= 75
        for count, (red, green, blue) in colors
    )


@pytest.mark.parametrize("axis", [add_linear_step_axis, add_sqrt_step_axis])
def test_final_resolution_draws_text_after_stretching(axis):
    native = Image.new("RGB", (16, 8), "orange")
    # The result must be identical to an already stretched, unlabelled diagram:
    # stretching a labelled frame would distort the glyphs and fail this check.
    expected = axis(Image.new("RGB", (320, 360), "orange"), 99)
    actual = axis(native, 99, resolution=(320, 360))
    assert actual.size == (320, 360)
    assert actual.tobytes() == expected.tobytes()
    assert native.size == (16, 8)


def test_enlargement_preserves_sharp_cells_and_source_colors():
    source = Image.new("RGB", (2, 2), "white")
    source.putpixel((0, 0), (255, 165, 0))
    enlarged = _resize_diagram(source, (200, 200))
    assert set(enlarged.get_flattened_data()) == {(255, 255, 255), (255, 165, 0)}
    assert enlarged.getpixel((99, 99)) == (255, 165, 0)
    assert enlarged.getpixel((100, 99)) == (255, 255, 255)


def test_mixed_resize_averages_only_the_shrinking_axis():
    source = Image.new("RGB", (2, 2), "white")
    source.putpixel((0, 0), (0, 0, 0))
    resized = _resize_diagram(source, (1, 100))
    assert set(resized.get_flattened_data()) == {(128, 128, 128), (255, 255, 255)}
    assert resized.getpixel((0, 49)) == (128, 128, 128)
    assert resized.getpixel((0, 50)) == (255, 255, 255)


def test_adaptive_filter_keeps_large_early_cells_sharp():
    source = Image.new("RGB", (2, 2), "white")
    source.putpixel((0, 0), (255, 165, 0))
    assert _resize_diagram(source, (200, 200), "adaptive").tobytes() == (
        _resize_diagram(source, (200, 200), "cells").tobytes()
    )


def test_adaptive_filter_interpolates_dense_stripes_only():
    source = Image.new("RGB", (32, 32), "white")
    for column in range(0, 32, 2):
        for row in range(32):
            source.putpixel((column, row), (0, 0, 0))
    adaptive = _resize_diagram(source, (48, 48), "adaptive")
    assert len(set(adaptive.get_flattened_data())) > 2
    assert len(set(_resize_diagram(source, (48, 48)).get_flattened_data())) == 2
