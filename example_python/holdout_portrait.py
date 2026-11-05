"""4320x7680 poster matching the 100K video's snake sequence and reflections.

Run: python3 example_python/holdout_portrait.py [CACHE_ROOT]
Reflows the 32x19 video snake into a 16x38 portrait snake. Renders fresh at
268x174 diagram resolution, caching each original unreflected machine PNG.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    repo = Path(__file__).resolve().parents[1]
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '/mnt/m/deldir/the_600')
    video_manifest = root / '608_grid_snake_dark_upper_left_100000steps_animated_manifest.json'
    manifest = json.loads(video_manifest.read_text())
    assert manifest['steps'][-1] == 100000 and len(manifest['layout']) == 608
    with (repo / 'data/bb6_holdouts.tsv').open() as source:
        programs = {int(row['id']): row['program'] for row in csv.DictReader(source, delimiter='\t')}
    sequence = []
    for grid_row in range(19):
        section = manifest['layout'][grid_row * 32:(grid_row + 1) * 32]
        sequence.extend(section if grid_row % 2 == 0 else section[::-1])
    assert sorted(row['id'] for row in sequence) == list(range(1, 609))
    subprocess.run(['cargo', 'run', '--offline', '--release', '--example', 'holdout_stills', '--',
                    str(repo / 'data/bb6_holdouts.txt'), str(root), '100000', '268', '174'], cwd=repo, check=True)
    frame = Image.new('RGB', (4320, 7680), (12, 17, 24))
    draw = ImageDraw.Draw(frame)
    font_path = str(repo / 'examples/fonts/FiraCode-Regular.ttf')
    title_font, label_font = ImageFont.truetype(font_path, 42), ImageFont.truetype(font_path, 24)
    draw.text((12, 8), '608 Turing machines | 100,000 steps | snake order | fixed reflections', font=title_font, fill='white')
    layout = []
    for sequence_index, machine in enumerate(sequence):
        grid_row, snake_column = divmod(sequence_index, 16)
        grid_column = snake_column if grid_row % 2 == 0 else 15 - snake_column
        program = programs[machine['id']]
        source = root / program / '268x174/steps100000_binning_teal_v1/step_100000.png'
        with Image.open(source) as raw:
            diagram = raw.convert('RGB')
        if machine['display_flipped']:
            diagram = ImageOps.mirror(diagram)
        diagram = diagram.resize((268, 174), Image.Resampling.NEAREST)
        left, top = grid_column * 270, 80 + grid_row * 200
        frame.paste(diagram, (left, top + 25))
        draw.text((left + 4, top - 2), str(machine['id']), font=label_font, fill=(185, 195, 205))
        layout.append(dict(sequence_index=sequence_index + 1, grid_row=grid_row + 1,
                           grid_column=grid_column + 1, **machine, program=program))
    output = root / '608_snake_100000steps_portrait_4320x7680.png'
    temporary = output.with_suffix('.tmp.png')
    frame.save(temporary, dpi=(300, 300))
    temporary.replace(output)
    frame.resize((1080, 1920), Image.Resampling.LANCZOS).save(root / '608_snake_100000steps_portrait_preview.png')
    with output.with_suffix('.tsv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(layout[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(layout)
    output.with_suffix('.json').write_text(json.dumps({
        'dimensions': [4320, 7680], 'columns': 16, 'rows': 38, 'steps': 100000,
        'source_video_manifest': str(video_manifest), 'machines': 608,
        'order': 'same linear snake traversal as video, reflowed to 16 columns',
        'reflections': 'exact video reflections, not recomputed at 100K',
        'rendering': 'fresh binned diagrams at 268x174, nearest resized to fit; original palette',
    }, indent=2) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
