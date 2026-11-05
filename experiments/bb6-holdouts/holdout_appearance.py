"""Compare appearance-only arrangements of the 608 cached 10K diagrams.

Run with Python containing numpy, scipy, and Pillow:
    python experiments/bb6-holdouts/holdout_appearance.py CACHE_ROOT
Writes contact sheets plus reusable row-major layout TSVs. No source categories
are read. Horizontal reflection is ignored in distances, never in display.
"""

import csv
import json
from pathlib import Path

from experiment import EXPERIMENT, cache_root

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, squareform


def normalized(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def mirror_distance(features, reflected):
    features, reflected = normalized(features), normalized(reflected)
    similarity = np.maximum(features @ features.T, features @ reflected.T)
    distance = np.clip(1 - similarity, 0, 2)
    blank = np.linalg.norm(features, axis=1) == 0
    distance[np.ix_(blank, blank)] = 0
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    return distance


def neighbors(position):
    row, column = divmod(position, 32)
    return [other for other in (position - 32 if row else -1,
                               position + 32 if row < 18 else -1,
                               position - 1 if column else -1,
                               position + 1 if column < 31 else -1) if other >= 0]


NEIGHBORS = [neighbors(position) for position in range(608)]
EDGES = [(position, other) for position in range(608)
         for other in NEIGHBORS[position] if other > position]


def neighbor_cost(layout, distance):
    return float(np.mean([distance[layout[first], layout[second]] for first, second in EDGES]))


def arrange_2d(distance):
    # Classical MDS: approximate the supplied dissimilarities in two dimensions.
    squared = distance ** 2
    gram = -0.5 * (squared - squared.mean(axis=0)[None, :]
                   - squared.mean(axis=1)[:, None] + squared.mean())
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    coordinates = eigenvectors[:, -2:] * np.sqrt(np.maximum(eigenvalues[-2:], 0))
    # Largest-variance axis runs across the wider grid.
    coordinates = coordinates[:, ::-1]
    for axis in range(2):
        extreme = np.argmax(np.abs(coordinates[:, axis]))
        if coordinates[extreme, axis] < 0:
            coordinates[:, axis] *= -1
    coordinates = (coordinates - coordinates.min(axis=0)) / np.maximum(np.ptp(coordinates, axis=0), 1e-12)
    slots = np.array([(column / 31, row / 18) for row in range(19) for column in range(32)])
    machines, positions = linear_sum_assignment(cdist(coordinates, slots, metric='sqeuclidean'))
    layout = np.empty(608, dtype=int)
    layout[positions] = machines
    initial = neighbor_cost(layout, distance)
    # Greedy swaps improve the actual four-neighbor grid objective.
    random = np.random.default_rng(600)
    for first, second in random.integers(0, 608, size=(150000, 2)):
        if first == second:
            continue
        affected = {(min(position, other), max(position, other))
                    for position in (first, second) for other in NEIGHBORS[position]}
        before = sum(distance[layout[left], layout[right]] for left, right in affected)
        layout[first], layout[second] = layout[second], layout[first]
        after = sum(distance[layout[left], layout[right]] for left, right in affected)
        if after >= before:
            layout[first], layout[second] = layout[second], layout[first]
    return layout, {'before_swaps': initial, 'after_swaps': neighbor_cost(layout, distance)}


def main():
    repo = Path(__file__).resolve().parents[2]
    root = cache_root()
    destination = root / 'appearance/10k_mirror_cosine_v1'
    destination.mkdir(parents=True, exist_ok=True)
    with (EXPERIMENT / 'data/bb6_holdouts.tsv').open() as source:
        # Deliberately take only identifiers and program strings.
        rows = [{'id': int(row['id']), 'program': row['program']} for row in csv.DictReader(source, delimiter='\t')]
    assert [row['id'] for row in rows] == list(range(1, 609))
    tiles, coarse, fine, last = [], [], [], []
    for row in rows:
        path = root / row['program'] / '1920x984/steps10000_frames3_binning_v1/step_10000.png'
        with Image.open(path) as source:
            image = source.convert('RGB')
            tiles.append(image.resize((118, 92), Image.Resampling.BOX))
            green = image.getchannel('G')
            for size, output in ((16, coarse), (64, fine)):
                output.append(np.clip((np.asarray(green.resize((size, size), Image.Resampling.BOX), dtype=float) - 17) / 201, 0, 1))
            bottom = green.crop((0, green.height - 1, green.width, green.height))
            last.append(np.clip((np.asarray(bottom.resize((64, 1), Image.Resampling.BOX), dtype=float) - 17) / 201, 0, 1))
        if row['id'] % 100 == 0:
            print(f"Loaded {row['id']}/608 diagrams", flush=True)
    def whole_features(mirror):
        parts = []
        for values in (np.array(coarse), np.array(fine)):
            if mirror:
                values = values[:, :, ::-1]
            parts.append(normalized(values.reshape(608, -1)))
        return np.concatenate(parts, axis=1)
    whole_distance = mirror_distance(whole_features(False), whole_features(True))
    last = np.array(last)
    row_distance = mirror_distance(last.reshape(608, -1), last[:, :, ::-1].reshape(608, -1))
    np.savez_compressed(destination / 'distances.npz', whole=whole_distance, last_row=row_distance)
    condensed = squareform(whole_distance, checks=True)
    tree = linkage(condensed, method='average')
    sequence = leaves_list(optimal_leaf_ordering(tree, condensed))
    snake = np.empty(608, dtype=int)
    for grid_row in range(19):
        section = sequence[grid_row * 32:(grid_row + 1) * 32]
        snake[grid_row * 32:(grid_row + 1) * 32] = section if grid_row % 2 == 0 else section[::-1]
    print('Linear ordering ready; arranging 2D layouts', flush=True)
    whole_layout, whole_stats = arrange_2d(whole_distance)
    last_layout, last_stats = arrange_2d(row_distance)
    layouts = [
        ('00_original_index', 'Original index | baseline', np.arange(608)),
        ('01_whole_image_snake', 'Whole image | cosine + clustering | snake order', snake),
        ('02_whole_image_2d', 'Whole image | cosine | 2D arrangement', whole_layout),
        ('03_last_row_2d', 'Final rendered row only | cosine | 2D arrangement', last_layout),
    ]
    font_path = str(repo / 'examples/fonts/FiraCode-Regular.ttf')
    font, header_font = ImageFont.truetype(font_path, 16), ImageFont.truetype(font_path, 24)
    statistics = {}
    for name, title, layout in layouts:
        assert sorted(layout.tolist()) == list(range(608))
        frame = Image.new('RGB', (3840, 2160), (12, 17, 24))
        draw = ImageDraw.Draw(frame)
        draw.text((8, 0), title + ' | 10K steps | reflection ignored; original orientation shown', font=header_font, fill='white')
        for position, machine in enumerate(layout):
            left, top = (position % 32) * 120, 32 + (position // 32) * 112
            frame.paste(tiles[machine], (left, top + 18))
            draw.text((left + 4, top - 2), str(rows[machine]['id']), font=font, fill=(185, 195, 205))
        frame.save(destination / f'{name}_4k.png')
        frame.resize((1920, 1080), Image.Resampling.LANCZOS).save(destination / f'{name}_hd.png')
        with (destination / f'{name}.tsv').open('w', newline='') as output:
            writer = csv.writer(output, delimiter='\t')
            writer.writerow(['position', 'grid_row', 'grid_column', 'id', 'program'])
            for position, machine in enumerate(layout):
                writer.writerow([position + 1, position // 32 + 1, position % 32 + 1, rows[machine]['id'], rows[machine]['program']])
        statistics[name] = {'whole_image_neighbor_distance': neighbor_cost(layout, whole_distance),
                            'last_row_neighbor_distance': neighbor_cost(layout, row_distance)}
        print(f'Saved {name}', flush=True)
    # Compact 2x2 comparison, preserving the sheets' 16:9 proportions.
    comparison = Image.new('RGB', (1920, 1080))
    for position, (name, _, _) in enumerate(layouts):
        with Image.open(destination / f'{name}_hd.png') as sheet:
            comparison.paste(sheet.resize((960, 540), Image.Resampling.LANCZOS),
                             ((position % 2) * 960, (position // 2) * 540))
    comparison.save(destination / 'comparison.png')
    (destination / 'manifest.json').write_text(json.dumps({
        'machines': 608, 'steps': 10000, 'seed': 600, 'categories_used': False,
        'representation': 'foreground intensity from green channel, background subtracted; per-image tape/time extent auto-fitted',
        'whole_features': 'equal-weight normalized 16x16 and 64x64 BOX images',
        'last_row_features': 'bottom row of cached binned diagram, BOX resized to 64; not exact final tape',
        'distance': 'minimum cosine distance under horizontal reflection',
        'linear': 'average-linkage hierarchical clustering + optimal leaf ordering, laid out as a snake',
        'two_dimensional': 'classical MDS, Hungarian assignment to grid, 150000 seeded greedy neighbor swaps',
        'whole_2d_optimization': whole_stats, 'last_row_2d_optimization': last_stats,
        'neighbor_scores': statistics,
        'limits': '2D embedding is approximate. Neighbor scores are comparable across layouts only within the same feature metric. Display always retains original orientation. No semantic labels or machine dynamics beyond cached pixels used.',
    }, indent=2) + '\n')
    print(json.dumps(statistics, indent=2))
    print(destination, flush=True)


if __name__ == '__main__':
    main()
