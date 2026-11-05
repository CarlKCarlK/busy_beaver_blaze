"""Render an 8K landscape version of the 100K video's final frame.

Run: python3 experiments/bb6-holdouts/holdout_landscape.py CACHE_ROOT
Keeps the video's exact 32x19 placement, reflections, palette and header.
Renders diagrams at 4x the video tile resolution rather than enlarging pixels.
"""

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

from experiment import EXPERIMENT, external_directory

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache_root', type=external_directory)
    parser.add_argument('--output-dir', type=external_directory)
    options = parser.parse_args()
    root = options.cache_root
    output_root = options.output_dir or root
    output_root.mkdir(parents=True, exist_ok=True)
    source_manifest = EXPERIMENT / 'layouts/snake_100k.json'
    manifest = json.loads(source_manifest.read_text())
    layout = manifest['layout']
    assert manifest['steps'][-1] == 100000 and len(layout) == 608
    assert sorted(row['id'] for row in layout) == list(range(1, 609))
    with (EXPERIMENT / 'data/bb6_holdouts.tsv').open() as source:
        programs = {int(row['id']): row['program'] for row in csv.DictReader(source, delimiter='\t')}
    subprocess.run(['cargo', 'run', '--offline', '--release', '--example', 'holdout_stills', '--',
                    str(EXPERIMENT / 'data/bb6_holdouts.txt'), str(root), '100000', '236', '184'], cwd=repo, check=True)
    font_path = str(repo / 'examples/fonts/FiraCode-Regular.ttf')
    font = ImageFont.truetype(font_path, 26)
    notation_font = ImageFont.truetype(font_path, 12)
    header_font = ImageFont.truetype(font_path, 48)
    background = Image.new('L', (7680, 4320), 0)
    labels = ImageDraw.Draw(background)
    for position, row in enumerate(layout):
        left, top = (position % 32) * 240, 64 + (position // 32) * 224
        prefix = f"{row['id']}/"
        labels.text((left + 8, top + 3), prefix, font=font, fill=185, anchor='lt')
        notation_left = left + math.ceil(12 + labels.textlength(prefix, font=font))
        groups = programs[row['id']].split('_')
        notation_lines = ['_'.join(groups[:3]) + '_', '_'.join(groups[3:])]
        assert all(notation_left + labels.textlength(line, font=notation_font) <= left + 236
                   for line in notation_lines), 'Notation must fit inside its tile'
        for line_index, line in enumerate(notation_lines):
            labels.text((notation_left, top + 3 + 14 * line_index), line,
                        font=notation_font, fill=222, anchor='lt')
    frame = ImageOps.colorize(background, (12, 17, 24), (77, 218, 210))
    for position, row in enumerate(layout):
        program = programs[row['id']]
        source = root / program / '236x184/steps100000_binning_teal_v1/step_100000.png'
        with Image.open(source) as raw:
            diagram = raw.convert('RGB')
        if row['display_flipped']:
            diagram = ImageOps.mirror(diagram)
        tile = diagram.resize((236, 184), Image.Resampling.NEAREST)
        left, top = (position % 32) * 240, 64 + (position // 32) * 224
        frame.paste(tile, (left, top + 36))
    ImageDraw.Draw(frame).text((16, 0),
        '608 Turing machines | snake order, fixed reflections | 100,000 / 100,000 steps | frame 1800 / 1800',
        font=header_font, fill='white')
    output = output_root / '608_grid_snake_dark_upper_left_100000steps_final_7680x4320_notation.png'
    temporary = output.with_suffix('.tmp.png')
    frame.save(temporary, dpi=(300, 300))
    temporary.replace(output)
    frame.resize((1920, 1080), Image.Resampling.LANCZOS).save(output.with_name(output.stem + '_preview.png'))
    output.with_suffix('.json').write_text(json.dumps({
        'source_video_manifest': str(source_manifest), 'dimensions': [7680, 4320],
        'scale': 4, 'steps': 100000, 'columns': 32, 'rows': 19,
        'layout': layout, 'tile_dimensions': [236, 184],
        'labels': 'original list ID / original standard TM notation; notation wraps after third state; original notation is retained for reflected displays',
        'rendering': 'fresh 4x-resolution binned diagrams, exact video placement/reflections and scaled labels/header',
    }, indent=2) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
