"""Parallel preparation must preserve pixels, order, and bounded slot ownership."""

import subprocess
import time

import pytest
from PIL import Image

from busy_beaver_blaze.movie_encoding import ordered_rgb_frames, video_encoder_args


def render_test_frame(index):
    # Encourage workers to finish out of order while the consumer expects order.
    time.sleep(0.005 * (7 - index))
    return Image.new("RGB", (4, 3), (index, index * 2, index * 3))


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_frames_preserve_order_and_pixels(workers):
    outputs = [bytes(frame) for frame in ordered_rgb_frames(list(range(8)), render_test_frame, workers, 36)]
    assert outputs == [bytes((index, index * 2, index * 3)) * 12 for index in range(8)]


def test_encoder_selection_and_cpu_fallback(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0))
    name, options = video_encoder_args("auto", 16)
    assert name == "NVIDIA NVENC"
    assert options[options.index("-c:v") + 1] == "h264_nvenc"
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1, stderr=b"no GPU"),
    )
    name, options = video_encoder_args("auto", 16)
    assert name == "CPU x264"
    assert options[options.index("-threads") + 1] == "16"
    with pytest.raises(RuntimeError, match="no GPU"):
        video_encoder_args("nvenc", 16)
