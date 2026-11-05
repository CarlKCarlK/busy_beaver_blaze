"""Validate the frozen dataset, identity mapping, snake layout and time schedule.

Run: python3 experiments/bb6-holdouts/validate.py
Uses only the standard library; creates no files and runs no simulations.
"""

import csv
import json
import re

from experiment import EXPERIMENT


def main():
    with (EXPERIMENT / 'data/bb6_holdouts.tsv').open() as source:
        machines = list(csv.DictReader(source, delimiter='\t'))
    assert [int(machine['id']) for machine in machines] == list(range(1, 609))
    programs = [machine['program'] for machine in machines]
    assert len(set(programs)) == 608
    assert programs == (EXPERIMENT / 'data/bb6_holdouts.txt').read_text().splitlines()
    pattern = re.compile(r'(?:[01][LR][A-F]|---){2}(?:_(?:[01][LR][A-F]|---){2}){5}')
    assert all(pattern.fullmatch(program) for program in programs)
    provenance = json.loads((EXPERIMENT / 'data/bb6_holdouts.source.json').read_text())
    assert provenance['list']['count'] == 608
    assert provenance['list']['published'] == '2026-10-30'
    layout = json.loads((EXPERIMENT / 'layouts/snake_100k.json').read_text())
    assert layout['machines'] == 608 and (layout['columns'], layout['rows']) == (32, 19)
    assert sorted(machine['id'] for machine in layout['layout']) == list(range(1, 609))
    with (EXPERIMENT / 'layouts/snake.tsv').open() as source:
        positions = list(csv.DictReader(source, delimiter='\t'))
    assert len(positions) == 608
    for position, (machine, row) in enumerate(zip(layout['layout'], positions)):
        assert machine['id'] == int(row['id'])
        assert type(machine['display_flipped']) is bool
        assert row['display_flipped'] in ('true', 'false')
        assert machine['display_flipped'] == (row['display_flipped'] == 'true')
        assert row['program'] == programs[machine['id'] - 1]
        reflected = row['program'].translate(str.maketrans('LR', 'RL'))
        assert row['display_program'] == (reflected if machine['display_flipped'] else row['program'])
        assert int(row['position']) == position + 1
        assert (int(row['grid_row']), int(row['grid_column'])) == (position // 32 + 1, position % 32 + 1)
    assert layout['steps'] == [1 + frame * 99999 // 1799 for frame in range(1800)]
    assert len(set(layout['steps'])) == 1800
    assert layout['frames'] == 1800 and layout['fps'] == 30 and layout['duration_seconds'] == 60
    print('Validated 608 unique programs, original IDs, exact snake/reflection mapping and 1800 targets ending at 100K.')


if __name__ == '__main__':
    main()
