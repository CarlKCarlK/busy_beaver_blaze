"""Sequential Square Root movie frames through the public Python API."""

from io import BytesIO

import pytest
from PIL import Image

from busy_beaver_blaze import SqrtPngDataIterator, create_frame, log_step_iterator


@pytest.mark.parametrize("pixel_policy", ["binning", "sampling"])
def test_movie_frames(pixel_policy):
    steps = log_step_iterator(130, 12)
    frames = list(SqrtPngDataIterator(
        steps, program="1RA2RA0RA", resolution=(16, 8),
        pixel_policy=pixel_policy, colors=["#fff", "#f00", "#00f"],
    ))
    assert [step_index for step_index, _ in frames] == steps
    for step_index, png in frames:
        assert isinstance(png, bytes)
        image = Image.open(BytesIO(png))
        image.load()
        assert image.format == "PNG"
        frame = create_frame(png, "", step_index, (320, 180))
        assert frame.size == (320, 180)


def test_duplicate_steps_and_exhaustion():
    frames = SqrtPngDataIterator([0, 0, 8, 8], program="1RA1RA", resolution=(8, 4))
    assert iter(frames) is frames
    results = list(frames)
    assert [step_index for step_index, _ in results] == [0, 0, 8, 8]
    assert results[0] == results[1]
    assert results[2] == results[3]
    assert list(frames) == []


def test_halt_does_not_emit_unreached_steps():
    frames = SqrtPngDataIterator([0, 1, 100], program="1RZ1RZ", resolution=(8, 4))
    assert [step_index for step_index, _ in frames] == [0]
    assert list(frames) == []


@pytest.mark.parametrize("kwargs, message", [
    ({"frame_steps": []}, "cannot be empty"),
    ({"frame_steps": [10, 1]}, "monotonically increasing"),
    ({"resolution": (1, 8)}, "Image goals"),
    ({"resolution": (8, 0)}, "Image goals"),
    ({"program": "invalid"}, ".+"),
    ({"pixel_policy": "invalid"}, "Invalid pixel_policy"),
    ({"colors": ["#fff"]}, ".+"),
    ({"colors": ["#fff", "#ggg"]}, "Invalid"),
])
def test_invalid_arguments(kwargs, message):
    arguments = dict(frame_steps=[0], program="1RA1RA", resolution=(8, 4))
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SqrtPngDataIterator(**arguments)
