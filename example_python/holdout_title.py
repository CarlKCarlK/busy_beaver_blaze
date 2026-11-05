"""Add a poster title to the existing 8K notation sheet without rerendering.

Run: python3 example_python/holdout_title.py [CACHE_ROOT]
Replaces the old 64-pixel video header with a 576-pixel title band.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    repo = Path(__file__).resolve().parents[1]
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '/mnt/m/deldir/the_600')
    source = root / '608_grid_snake_dark_upper_left_100000steps_final_7680x4320_notation.png'
    with Image.open(source) as original:
        assert original.size == (7680, 4320)
        diagrams = original.crop((0, 64, 7680, 4320)).convert('RGB')
    frame = Image.new('RGB', (7680, diagrams.height + 576), (12, 17, 24))
    frame.paste(diagrams, (0, 576))
    draw = ImageDraw.Draw(frame)
    font_path = str(repo / 'examples/fonts/FiraCode-Regular.ttf')
    lines = [
        ('608 BB(6) Holdouts', 200, 32, 'white'),
        ('100,000 steps, in rough similarity order', 100, 260, (77, 218, 210)),
        ('https://denniskoshta.com/bbchallenge/bb6/', 56, 392, (185, 195, 205)),
        ('https://github.com/CarlKCarlK/busy_beaver_blaze', 56, 472, (185, 195, 205)),
    ]
    for text, size, top, color in lines:
        font = ImageFont.truetype(font_path, size)
        width = draw.textlength(text, font=font)
        assert width < frame.width - 128
        draw.text(((frame.width - width) / 2, top), text, font=font, fill=color, anchor='lt')
    output = root / '608_BB6_Holdouts_100000steps_titled_8k.png'
    frame.save(output, dpi=(300, 300))
    # Verify the diagram region is unchanged, pixel for pixel.
    with Image.open(output) as saved:
        assert saved.crop((0, 576, 7680, frame.height)).tobytes() == diagrams.tobytes()
    frame.resize((1920, 1208), Image.Resampling.LANCZOS).save(root / '608_BB6_Holdouts_100000steps_titled_preview.png')
    print(f'{output} ({frame.width}x{frame.height}); diagram pixels unchanged', flush=True)


if __name__ == '__main__':
    main()
