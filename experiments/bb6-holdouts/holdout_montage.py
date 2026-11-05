"""Quick 60.8-second HD montage, in original holdout index order.

Run from the repository root:
    python3 experiments/bb6-holdouts/holdout_montage.py

Requires Pillow, ffmpeg, and the repository Rust toolchain. Frames are reused
by machine notation, dimensions, step count, palette/layout version and sampling
settings. Delete a machine's cache directory to force that machine to rerender.
"""

import csv
import json
import subprocess
from pathlib import Path

from experiment import EXPERIMENT, cache_root

from PIL import Image, ImageDraw, ImageFont


def main():
    repo = Path(__file__).resolve().parents[2]
    root = cache_root()
    rows = list(csv.DictReader((EXPERIMENT / 'data/bb6_holdouts.tsv').open(), delimiter='\t'))
    assert [int(row['id']) for row in rows] == list(range(1, 609))
    subprocess.run(['cargo', 'run', '--offline', '--release', '--example', 'holdout_frames', '--',
                    str(EXPERIMENT / 'data/bb6_holdouts.txt'), str(root)], cwd=repo, check=True)
    font = ImageFont.truetype(str(repo / 'examples/fonts/FiraCode-Regular.ttf'), 25)
    title_font = ImageFont.truetype(str(repo / 'examples/fonts/FiraCode-Regular.ttf'), 34)
    frame_paths = []
    for row in rows:
        for step in (3333, 6666, 10000):
            folder = root / row['program'] / '1920x1080' / 'steps10000_frames3_binning_labels_v2'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"index_{int(row['id']):03}_step_{step:05}.png"
            if not path.exists():
                raw = root / row['program'] / '1920x984' / 'steps10000_frames3_binning_v1' / f'step_{step:05}.png'
                frame = Image.new('RGB', (1920, 1080), (12, 17, 24))
                with Image.open(raw) as diagram:
                    frame.paste(diagram.convert('RGB').resize((1920, 984), Image.Resampling.NEAREST), (0, 96))
                draw = ImageDraw.Draw(frame)
                name = f"  {row['name']}" if row['name'] else ''
                draw.text((28, 8), f"#{int(row['id']):03} / 608{name}", font=title_font, fill='white')
                draw.text((1450, 15), f'{step:,} / 10,000 steps', font=font, fill=(77, 218, 210))
                draw.text((28, 55), row['program'], font=font, fill=(185, 195, 205))
                temporary = path.with_suffix('.tmp.png')
                frame.save(temporary, compress_level=1)
                temporary.replace(path)
            frame_paths.append(path)
        if int(row['id']) % 100 == 0:
            print(f"Composited {row['id']}/608", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'montage_index.ffconcat'
    # All generated machine paths contain only standard TM notation; root is
    # escaped for ffconcat's single-quoted filename syntax.
    def quote(path):
        return str(path.absolute()).replace("'", "'\\''")
    manifest.write_text('ffconcat version 1.0\n' + ''.join(
        f"file '{quote(path)}'\nduration 0.04\n" for path in frame_paths
    ))
    output = root / '608_index_10000steps_60.8s.mp4'
    temporary = root / '608_index_10000steps_60.8s.tmp.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
                    '-r', '30', '-f', 'concat', '-safe', '0', '-i', str(manifest),
                    '-r', '30', '-frames:v', str(len(frame_paths)),
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)], check=True)
    temporary.replace(output)
    (root / 'montage_manifest.json').write_text(json.dumps({
        'machines': 608, 'order': 'original 1-based index', 'steps': [3333, 6666, 10000],
        'fps': 30, 'frames': len(frame_paths), 'duration_seconds': 60.8,
        'dimensions': [1920, 1080], 'palette': ['#0c1118', '#4ddad2'],
        'source': 'experiments/bb6-holdouts/data/bb6_holdouts.tsv', 'categories_used': False,
        'renderer': 'SpaceByTimeMachine, binning enabled; each frame auto-fits its tape/time extent',
    }, indent=2) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
