"""Bounded parallel frame preparation and selectable FFmpeg encoders."""

import argparse
import multiprocessing
import os
import subprocess
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Iterator

from PIL import Image

_worker_memory: SharedMemory | None = None
_worker_renderer: Callable[[Any], Image.Image] | None = None
_worker_frame_size = 0


def add_encoding_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose separate controls for image preparation and video compression."""
    processors = os.cpu_count() or 1
    parser.add_argument("--workers", type=int, default=min(8, max(1, processors // 4)))
    parser.add_argument("--encoder", choices=["auto", "cpu", "nvenc"], default="auto")
    parser.add_argument("--cpu-threads", type=int, default=max(1, processors // 2))


def video_encoder_args(encoder: str, cpu_threads: int) -> tuple[str, list[str]]:
    """Probe NVENC before selecting it; keep a CPU fallback for other machines."""
    assert cpu_threads > 0
    assert encoder in ("auto", "cpu", "nvenc")
    if encoder != "cpu":
        probe = subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "color=s=1280x720:r=30", "-frames:v", "1", "-an",
            "-c:v", "h264_nvenc", "-preset", "p5", "-f", "null", "-",
        ], capture_output=True, timeout=30, check=False)
        if probe.returncode == 0:
            return "NVIDIA NVENC", [
                "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "18", "-b:v", "0",
            ]
        if encoder == "nvenc":
            raise RuntimeError(f"NVENC is unavailable: {probe.stderr.decode(errors='replace')}")
        print("NVENC unavailable; using CPU encoding", flush=True)
    return "CPU x264", [
        "-c:v", "libx264", "-threads", str(cpu_threads), "-preset", "fast", "-crf", "18",
    ]


def _initialize_worker(name: str, frame_size: int, renderer: Callable[[Any], Image.Image]) -> None:
    global _worker_memory, _worker_renderer, _worker_frame_size
    _worker_memory = SharedMemory(name=name)
    _worker_renderer = renderer
    _worker_frame_size = frame_size


def _render_shared(task: Any, slot: int) -> None:
    assert _worker_memory is not None and _worker_renderer is not None
    frame = _worker_renderer(task)
    assert frame.mode == "RGB"
    data = frame.tobytes()
    assert len(data) == _worker_frame_size
    start = slot * _worker_frame_size
    buffer = _worker_memory.buf
    assert buffer is not None
    buffer[start:start + _worker_frame_size] = data


def ordered_rgb_frames(
    tasks: list[Any], renderer: Callable[[Any], Image.Image], workers: int, frame_size: int,
) -> Iterator[bytes | memoryview]:
    """Prepare frames concurrently in bounded shared-memory slots, yielding in order.

    A yielded memoryview is valid only until the next iteration. Copy the final
    frame if it must be retained for a hold. No frame files or pickled RGB buffers
    are created; workers share at most one RGB slot each.
    """
    assert workers > 0 and frame_size > 0
    if workers == 1:
        for task in tasks:
            data = renderer(task).tobytes()
            assert len(data) == frame_size
            yield data
        return
    if not tasks:
        return
    slots = min(workers, len(tasks))
    memory = SharedMemory(create=True, size=slots * frame_size)
    buffer = memory.buf
    assert buffer is not None
    try:
        with ProcessPoolExecutor(
            max_workers=slots, mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker, initargs=(memory.name, frame_size, renderer),
        ) as pool:
            pending = deque(pool.submit(_render_shared, tasks[index], index) for index in range(slots))
            for index in range(len(tasks)):
                pending.popleft().result()
                slot = index % slots
                view = buffer[slot * frame_size:(slot + 1) * frame_size]
                try:
                    yield view
                finally:
                    view.release()
                next_index = index + slots
                if next_index < len(tasks):
                    pending.append(pool.submit(_render_shared, tasks[next_index], slot))
    finally:
        memory.close()
        memory.unlink()
