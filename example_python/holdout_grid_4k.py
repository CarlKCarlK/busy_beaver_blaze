"""Titled 4K, 60-second, 100K-step video matching the saved snake poster.

Run: python3 example_python/holdout_grid_4k.py [CACHE_ROOT]
Fresh 118x80 tiles, 1,800 frames per machine, same saved order/reflections.
"""

import csv
import json
import math
import mmap
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    started = time.monotonic()
    repo = Path(__file__).resolve().parents[1]
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '/mnt/m/deldir/the_600')
    source_manifest = root / '608_grid_snake_dark_upper_left_100000steps_animated_manifest.json'
    original = json.loads(source_manifest.read_text())
    layout = original['layout']
    assert original['steps'][-1] == 100000
    assert sorted(row['id'] for row in layout) == list(range(1, 609))
    with (repo / 'data/bb6_holdouts.tsv').open() as source:
        programs = {int(row['id']): row['program'] for row in csv.DictReader(source, delimiter='\t')}
    print('Stage 1/3: rendering 608 tile caches at 118x80, through 100K steps', flush=True)
    subprocess.run(['cargo', 'run', '--offline', '--release', '--example', 'holdout_grid_frames', '--',
                    str(repo / 'data/bb6_holdouts.txt'), str(root), '100000', '118', '80'], cwd=repo, check=True)
    print('Stage 2/3: composing and encoding 1,800 titled 4K frames', flush=True)
    font_path = str(repo / 'examples/fonts/FiraCode-Regular.ttf')
    label_font = ImageFont.truetype(font_path, 13)
    notation_font = ImageFont.truetype(font_path, 6)
    counter_font = ImageFont.truetype(font_path, 20)
    background = Image.new('L', (3840, 2160), 0)
    labels = ImageDraw.Draw(background)
    positions = []
    for position, machine in enumerate(layout):
        left, top = (position % 32) * 120, 250 + (position // 32) * 100
        prefix = f"{machine['id']}/"
        labels.text((left + 4, top + 1), prefix, font=label_font, fill=185, anchor='lt')
        notation_left = left + math.ceil(6 + labels.textlength(prefix, font=label_font))
        groups = programs[machine['id']].split('_')
        for line_index, text in enumerate(('_'.join(groups[:3]) + '_', '_'.join(groups[3:]))):
            assert notation_left + labels.textlength(text, font=notation_font) <= left + 118
            labels.text((notation_left, top + 1 + 7 * line_index), text, font=notation_font, fill=222, anchor='lt')
        positions.append((left, top + 20))
    header = Image.new('RGB', (3840, 240), (12, 17, 24))
    title = ImageDraw.Draw(header)
    header_lines = [
        ('608 BB(6) Holdouts', 80, 10, 'white'),
        ('100,000 steps, in rough similarity order', 36, 104, (77, 218, 210)),
        ('https://denniskoshta.com/bbchallenge/bb6/', 22, 154, (185, 195, 205)),
        ('https://github.com/CarlKCarlK/busy_beaver_blaze', 22, 184, (185, 195, 205)),
    ]
    for text, size, top, color in header_lines:
        font = ImageFont.truetype(font_path, size)
        width = title.textlength(text, font=font)
        title.text(((3840 - width) / 2, top), text, font=font, fill=color, anchor='lt')
    output = root / '608_BB6_Holdouts_snake_100000steps_4k_60s.mp4'
    temporary = output.with_suffix('.tmp.mp4')
    cache_folder = '118x80/steps100000_frames1800_linear_binning_gray_v1'
    frame_bytes = 118 * 80
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '3840x2160',
               '-r', '30', '-i', 'pipe:0', '-frames:v', '1800',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-threads', '8',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    with ExitStack() as stack:
        caches = []
        for machine in layout:
            with (root / programs[machine['id']] / cache_folder / 'frames.gray').open('rb') as source:
                cache = stack.enter_context(mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ))
                assert len(cache) == 1800 * frame_bytes
                caches.append(cache)
        encoder = stack.enter_context(subprocess.Popen(command, stdin=subprocess.PIPE))
        encoding_started = time.monotonic()
        for frame_index in range(1800):
            gray_frame = background.copy()
            offset = frame_index * frame_bytes
            for machine, cache, position in zip(layout, caches, positions):
                tile = Image.frombytes('L', (118, 80), cache[offset:offset + frame_bytes])
                if machine['display_flipped']:
                    tile = ImageOps.mirror(tile)
                gray_frame.paste(tile, position)
            frame = ImageOps.colorize(gray_frame, (12, 17, 24), (77, 218, 210))
            frame.paste(header, (0, 0))
            step = 1 + frame_index * 99999 // 1799
            draw = ImageDraw.Draw(frame)
            counter = f'Step {step:,} / 100,000     Frame {frame_index + 1} / 1800'
            width = draw.textlength(counter, font=counter_font)
            draw.text(((3840 - width) / 2, 216), counter, font=counter_font, fill=(77, 218, 210), anchor='lt')
            if frame_index in (0, 899, 1799):
                frame.save(root / f'608_BB6_Holdouts_4k_frame_{frame_index + 1:04}.png')
            encoder.stdin.write(frame.tobytes())
            if (frame_index + 1) % 150 == 0:
                elapsed = time.monotonic() - encoding_started
                print(f'Encoded {frame_index + 1}/1800 frames; {elapsed:.0f}s encoding elapsed', flush=True)
        encoder.stdin.close()
        assert encoder.wait() == 0, '4K encoding failed'
    temporary.replace(output)
    print('Stage 3/3: verifying video dimensions, timing and frame count', flush=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration', '-of', 'json', str(output)]))
    video = probe['streams'][0]
    assert (video['width'], video['height']) == (3840, 2160)
    assert video['r_frame_rate'] == '30/1' and int(video['nb_frames']) == 1800
    assert abs(float(video['duration']) - 60) < 0.001
    output.with_suffix('.json').write_text(json.dumps({
        'dimensions': [3840, 2160], 'fps': 30, 'frames': 1800, 'duration_seconds': 60,
        'steps': [1 + index * 99999 // 1799 for index in range(1800)],
        'layout': layout, 'source_manifest': str(source_manifest), 'cache_folder': cache_folder,
        'tile_dimensions': [118, 80], 'header_height': 240,
        'labels': 'original index/original TM notation, wrapped into two lines',
        'encoding': {'codec': 'libx264', 'preset': 'veryfast', 'crf': 18, 'pixel_format': 'yuv420p'},
        'probe': video,
    }, indent=2) + '\n')
    print(f'Completed in {time.monotonic() - started:.0f}s: {output}', flush=True)


if __name__ == '__main__':
    main()
