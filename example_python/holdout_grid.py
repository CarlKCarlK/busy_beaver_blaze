"""Make an HD contact sheet and six-second grid video from the montage cache.

Run: python3 example_python/holdout_grid.py [CACHE_ROOT]
All 608 machines appear in index order in a 32-column, 19-row grid.
The video holds each cached snapshot (3333, 6666, 10000 steps) for two seconds.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    repo = Path(__file__).resolve().parents[1]
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '/mnt/m/deldir/the_600')
    with (repo / 'data/bb6_holdouts.tsv').open() as source:
        rows = list(csv.DictReader(source, delimiter='\t'))
    assert [int(row['id']) for row in rows] == list(range(1, 609))
    steps = (3333, 6666, 10000)
    raw_folder = '1920x984/steps10000_frames3_binning_v1'
    # Check all inputs before creating outputs. Run holdout_montage.py if missing.
    for row in rows:
        for step in steps:
            path = root / row['program'] / raw_folder / f'step_{step:05}.png'
            assert path.is_file(), f'Missing cached render: {path}'
    destination = root / 'grid/1920x1080/32x19_steps10000_frames3_box_v1'
    destination.mkdir(parents=True, exist_ok=True)
    font_path = str(repo / 'examples/fonts/FiraCode-Regular.ttf')
    font = ImageFont.truetype(font_path, 8)
    header_font = ImageFont.truetype(font_path, 12)
    paths = []
    for step in steps:
        path = destination / f'grid_step_{step:05}.png'
        if not path.exists():
            frame = Image.new('RGB', (1920, 1080), (12, 17, 24))
            draw = ImageDraw.Draw(frame)
            draw.text((4, 0), f'608 Turing machines | index order, left to right | {step:,} / 10,000 steps', font=header_font, fill='white')
            for position, row in enumerate(rows):
                left = (position % 32) * 60
                top = 16 + (position // 32) * 56
                source = root / row['program'] / raw_folder / f'step_{step:05}.png'
                with Image.open(source) as diagram:
                    tile = diagram.convert('RGB').resize((59, 46), Image.Resampling.BOX)
                    frame.paste(tile, (left, top + 9))
                draw.text((left + 2, top - 1), row['id'], font=font, fill=(185, 195, 205))
            temporary = path.with_suffix('.tmp.png')
            frame.save(temporary)
            temporary.replace(path)
        paths.append(path)
        print(f'Grid snapshot ready: {step:,} steps', flush=True)
    output = root / '608_grid_10000steps_6s.mp4'
    temporary = output.with_suffix('.tmp.mp4')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
               '-r', '0.5', '-f', 'image2pipe', '-vcodec', 'png', '-i', 'pipe:0',
               '-r', '30', '-frames:v', '180', '-c:v', 'libx264', '-preset', 'ultrafast',
               '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for path in paths:
            encoder.stdin.write(path.read_bytes())
        encoder.stdin.close()
        assert encoder.wait() == 0, 'Grid video encoding failed'
    temporary.replace(output)
    (destination / 'manifest.json').write_text(json.dumps({
        'machines': 608, 'columns': 32, 'rows': 19, 'dimensions': [1920, 1080],
        'order': 'original index, left to right then top to bottom',
        'steps': steps, 'seconds_per_snapshot': 2, 'video_duration_seconds': 6,
        'tile_dimensions': [60, 56], 'diagram_dimensions': [59, 46],
        'sampling': 'BOX downsampling of existing binned renders; each tape auto-fits',
        'categories_used': False, 'source': 'data/bb6_holdouts.tsv',
    }, indent=2) + '\n')
    print(f'Video: {output}\nContact sheet: {paths[-1]}', flush=True)


if __name__ == '__main__':
    main()
