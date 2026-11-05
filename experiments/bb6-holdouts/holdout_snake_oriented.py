"""Make the 4K appearance snake with darker upper halves facing left.

Run: python3 experiments/bb6-holdouts/holdout_snake_oriented.py CACHE_ROOT
Preserves the existing snake layout; records display reflections separately
from the original program strings and IDs.
"""

import csv
from pathlib import Path

from experiment import cache_root

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat


def main():
    repo = Path(__file__).resolve().parents[2]
    root = cache_root()
    folder = root / 'appearance/10k_mirror_cosine_v1'
    with (folder / '01_whole_image_snake.tsv').open() as source:
        rows = list(csv.DictReader(source, delimiter='\t'))
    assert sorted(int(row['id']) for row in rows) == list(range(1, 609))
    frame = Image.new('RGB', (3840, 2160), (12, 17, 24))
    draw = ImageDraw.Draw(frame)
    font_path = str(repo / 'examples/fonts/FiraCode-Regular.ttf')
    font = ImageFont.truetype(font_path, 16)
    header_font = ImageFont.truetype(font_path, 24)
    draw.text((8, 0), 'Whole-image snake | 10K steps | darker upper side faces left | original machine IDs', font=header_font, fill='white')
    for row in rows:
        path = root / row['program'] / '1920x984/steps10000_frames3_binning_v1/step_10000.png'
        with Image.open(path) as source:
            diagram = source.convert('RGB')
        sample = diagram.getchannel('G').resize((64, 64), Image.Resampling.BOX)
        upper_left = ImageStat.Stat(sample.crop((0, 0, 32, 32))).mean[0]
        upper_right = ImageStat.Stat(sample.crop((32, 0, 64, 32))).mean[0]
        flip = upper_left > upper_right
        if flip:
            diagram = ImageOps.mirror(diagram)
        row['display_flipped'] = str(flip).lower()
        row['orientation_rule'] = 'mean upper-left intensity <= mean upper-right, measured at 64x64'
        # Provide the executable reflected variant too; preserve the source program.
        row['display_program'] = row['program'].translate(str.maketrans('LR', 'RL')) if flip else row['program']
        tile = diagram.resize((118, 92), Image.Resampling.BOX)
        left = (int(row['grid_column']) - 1) * 120
        top = 32 + (int(row['grid_row']) - 1) * 112
        frame.paste(tile, (left, top + 18))
        draw.text((left + 4, top - 2), row['id'], font=font, fill=(185, 195, 205))
    output = folder / '01_whole_image_snake_dark_upper_left_4k.png'
    frame.save(output)
    frame.resize((1920, 1080), Image.Resampling.LANCZOS).save(folder / '01_whole_image_snake_dark_upper_left_hd.png')
    with (folder / '01_whole_image_snake_dark_upper_left.tsv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}; reflected {sum(row['display_flipped'] == 'true' for row in rows)}/608 diagrams")


if __name__ == '__main__':
    main()
