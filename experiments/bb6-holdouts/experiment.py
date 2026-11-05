"""Paths shared only by the opt-in BB(6) experiment scripts."""

import argparse
import __main__
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent
REPOSITORY = EXPERIMENT.parents[1]


def external_directory(value):
    """Keep generated media, cache files and local environments out of Git."""
    path = Path(value).expanduser().resolve()
    if path.is_relative_to(REPOSITORY):
        raise argparse.ArgumentTypeError('Choose a directory outside the repository')
    return path


def cache_root():
    parser = argparse.ArgumentParser(description=__main__.__doc__)
    parser.add_argument('cache_root', type=external_directory)
    return parser.parse_args().cache_root
