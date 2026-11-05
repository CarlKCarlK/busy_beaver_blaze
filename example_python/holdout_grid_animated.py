"""Render 1,800 distinct grid snapshots; 60s at 30fps.

Run: python3 example_python/holdout_grid_animated.py [CACHE_ROOT]
Optional: --layout LAYOUT.tsv --output-stem NAME --steps 100000
Layout positions and display_flipped flags stay fixed throughout the movie.
Caches each machine's 1,800 grayscale tiles in one frames.gray file (~4.9 MB),
avoiding a million tiny PNG files. Each tile is 59x46 bytes, row-major, ordered
by frame; step = 1 + frame_index * (steps - 1) // 1799. Complete caches reuse.
"""

import argparse
import csv
import json
import mmap
import subprocess
from contextlib import ExitStack
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache_root', nargs='?', default='/mnt/m/deldir/the_600')
    parser.add_argument('--layout', type=Path, help='Row-major layout TSV with original programs and optional display_flipped')
    parser.add_argument('--output-stem', default='608_grid')
    parser.add_argument('--steps', type=int, default=10000)
    arguments = parser.parse_args()
    assert arguments.steps >= 1800, 'Need at least 1800 steps for distinct targets'
    root = Path(arguments.cache_root)
    with (repo / 'data/bb6_holdouts.tsv').open() as source:
        rows = list(csv.DictReader(source, delimiter='\t'))
    assert [int(row['id']) for row in rows] == list(range(1, 609))
    if arguments.layout:
        originals = {row['id']: row['program'] for row in rows}
        with arguments.layout.open() as source:
            rows = list(csv.DictReader(source, delimiter='\t'))
        assert sorted(int(row['id']) for row in rows) == list(range(1, 609))
        for position, row in enumerate(rows):
            assert row['program'] == originals[row['id']]
            assert int(row['position']) == position + 1
            assert (int(row['grid_row']), int(row['grid_column'])) == (position // 32 + 1, position % 32 + 1)
            assert row.get('display_flipped', 'false') in ('true', 'false')
    reflections = [row.get('display_flipped', 'false') == 'true' for row in rows]
    order_label = 'snake order, fixed reflections' if arguments.layout else 'index order'
    subprocess.run(['cargo', 'run', '--offline', '--release', '--example',
                    'holdout_grid_frames', '--', str(repo / 'data/bb6_holdouts.txt'),
                    str(root), str(arguments.steps)], cwd=repo, check=True)
    font_path = str(repo / 'examples/fonts/FiraCode-Regular.ttf')
    font = ImageFont.truetype(font_path, 8)
    header_font = ImageFont.truetype(font_path, 12)
    background = Image.new('L', (1920, 1080), 0)
    labels = ImageDraw.Draw(background)
    positions = []
    for position, row in enumerate(rows):
        left, top = (position % 32) * 60, 16 + (position // 32) * 56
        labels.text((left + 2, top - 1), row['id'], font=font, fill=185)
        positions.append((left, top + 9))
    output_base = f'{arguments.output_stem}_{arguments.steps}steps'
    output = root / f'{output_base}_60s_animated.mp4'
    temporary = output.with_suffix('.tmp.mp4')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080',
               '-r', '30', '-i', 'pipe:0', '-frames:v', '1800',
               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    cache_folder = f'59x46/steps{arguments.steps}_frames1800_linear_binning_gray_v1'
    with ExitStack() as stack:
        caches = []
        for row in rows:
            with (root / row['program'] / cache_folder / 'frames.gray').open('rb') as source:
                cache = stack.enter_context(mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ))
                assert len(cache) == 1800 * 59 * 46
                caches.append(cache)
        encoder = stack.enter_context(subprocess.Popen(command, stdin=subprocess.PIPE))
        for frame_index in range(1800):
            frame = background.copy()
            offset = frame_index * 59 * 46
            for cache, position, reflected in zip(caches, positions, reflections):
                tile = Image.frombytes('L', (59, 46), cache[offset:offset + 59 * 46])
                if reflected:
                    tile = ImageOps.mirror(tile)
                frame.paste(tile, position)
            frame = ImageOps.colorize(frame, (12, 17, 24), (77, 218, 210))
            step = 1 + frame_index * (arguments.steps - 1) // 1799
            ImageDraw.Draw(frame).text((4, 0),
                f'608 Turing machines | {order_label} | {step:,} / {arguments.steps:,} steps | frame {frame_index + 1} / 1800',
                font=header_font, fill='white')
            encoder.stdin.write(frame.tobytes())
            if frame_index == 1799:
                frame.save(root / f'{output_base}_animated_final.png')
            if (frame_index + 1) % 150 == 0:
                print(f'Encoded {frame_index + 1}/1800 frames', flush=True)
        encoder.stdin.close()
        assert encoder.wait() == 0, 'Video encoding failed'
    temporary.replace(output)
    (root / f'{output_base}_animated_manifest.json').write_text(json.dumps({
        'frames': 1800, 'fps': 30, 'duration_seconds': 60, 'dimensions': [1920, 1080],
        'machines': 608, 'columns': 32, 'rows': 19, 'order': order_label,
        'layout_source': str(arguments.layout) if arguments.layout else None,
        'layout': [{'id': int(row['id']), 'display_flipped': reflected}
                   for row, reflected in zip(rows, reflections)],
        'steps': [1 + frame * (arguments.steps - 1) // 1799 for frame in range(1800)],
        'cache_folder': cache_folder, 'cache_format': '1800 row-major 59x46 uint8 grayscale frames',
        'rendering': 'independent simulation snapshots; linear time; binning; nearest resize to tile',
        'categories_used': False,
    }, indent=2) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
