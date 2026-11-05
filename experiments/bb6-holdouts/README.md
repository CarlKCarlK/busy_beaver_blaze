# BB(6) holdout visualization experiment

An opt-in reproduction of the November 5, 2026 poster and video: 608 six-state,
two-symbol Turing machines, through 100,000 steps, in rough visual-similarity
order. These tools use the existing native renderer. They do not change the
library, Python package installation, browser visualizer, presets or defaults.

- [Full-resolution poster and credits](https://github.com/CarlKCarlK/busy_beaver_blaze/releases/tag/bb6-holdout-visualizations-2026-11-05)
- [One-minute 4K video](https://youtu.be/ncqby532REQ)
- [Source holdout browser](https://denniskoshta.com/bbchallenge/bb6/)

## Historical state and identifiers

The dataset is the October 30, 2026 list, downloaded November 5, with 608
original 1-based list IDs. It is a snapshot, not a live list of unresolved
machines. Some entries already had proofs when the metadata was downloaded.
Use the original machine notation as the stable identifier rather than
assuming positions in future lists will match.

The exact unmodified prototype is preserved on branch `the_600` and annotated
tag [`bb6-holdouts-source-2026-11-05`](https://github.com/CarlKCarlK/busy_beaver_blaze/tree/bb6-holdouts-source-2026-11-05),
commit `8af1295`. That snapshot includes the earlier HD, three-frame and
portrait experiments plus the original PR-comment draft. The public image
release tag, `bb6-holdout-visualizations-2026-11-05`, predates that source
snapshot and points to the original library baseline.

`data/` contains the programs as TSV and one-per-line text plus source and
format metadata. Structured TSV columns contain JSON values. Read the TSV
with a CSV/TSV parser, then decode those columns as JSON.

`layouts/snake.tsv` and `layouts/snake_100k.json` freeze the exact 32-column,
19-row arrangement and reflection flags used by the published outputs. The
similarity ordering and darker-upper-left reflections were selected at 10K
steps and remain fixed at 100K. Labels retain the original program notation
even when a diagram is reflected. Published behavioral categories were not
used to arrange the diagrams.

## Setup

Run the following from the repository root. Use Python 3.11 or newer, the
repository's pinned Rust toolchain, and FFmpeg with `libx264` plus `ffprobe` on
`PATH`. The scripts use the repository's bundled Fira Code font.

Create an environment outside the repository, separate from the main Python
package. `requirements.lock.txt` records the versions used for validation;
`requirements.txt` records compatible dependency ranges and the Pillow
security minimum used by this repository.

```bash
uv venv --python 3.14 /tmp/bb6-venv
uv pip install --python /tmp/bb6-venv/bin/python -r experiments/bb6-holdouts/requirements.lock.txt
source /tmp/bb6-venv/bin/activate
python experiments/bb6-holdouts/validate.py
```

Pillow is needed for rendering and composition. NumPy and SciPy are only
needed for the optional appearance-analysis workflow. These experiment
dependencies are not added to the main package or Cargo manifest.

## Reproduce the final video and poster

Choose an explicit external cache/output directory. No machine-specific WSL
path is assumed, and paths inside the repository are rejected. Existing
complete cache entries are reused. The scripts write fixed output names, so
use another external directory to keep a separate run.

```bash
python experiments/bb6-holdouts/holdout_grid_4k.py /tmp/bb6-output
python experiments/bb6-holdouts/holdout_landscape.py /tmp/bb6-output
python experiments/bb6-holdouts/holdout_title.py /tmp/bb6-output
```

The output files are:

- `608_BB6_Holdouts_snake_100000steps_4k_60s.mp4`: 3840 × 2160, 30 fps,
  exactly 60 seconds, with title, URLs, notation labels and a live step counter.
- `608_grid_snake_dark_upper_left_100000steps_final_7680x4320_notation.png`:
  fresh 8K landscape diagrams in the video arrangement.
- `608_BB6_Holdouts_100000steps_titled_8k.png`: 7680 × 4832 poster. The title
  operation preserves the diagram pixels exactly and adds height for the title.

The poster matches the original HD video's layout and reflections. The final
4K video reserves a title band and uses shorter tiles; the diagram order and
reflection mapping are identical across them.

The standard schedule uses 1,800 distinct simulation targets:
`1 + frame_index * 99999 // 1799`, starting at step 1 and ending at 100,000.
Frames are genuine simulation snapshots. Every diagram auto-fits its tape
and time extent, so tiles do not share a physical tape scale.

Renderers use eight workers. The 4K frame cache occupies about 10.3 GB before
video and still-image outputs; the full pipeline needs additional free space.
Do not put caches or generated PNG/MP4 files in Git.

## Cache layout and validation

The animated grayscale cache is:

```text
CACHE_ROOT/<original TM notation>/118x80/steps100000_frames1800_linear_binning_gray_v1/frames.gray
```

It contains 1,800 consecutive row-major 118 × 80 uint8 grayscale frames, with
no header. The frame size and total length are checked before use. The palette
is `#0c1118` to `#4ddad2`; reflection is applied during composition, keeping
the machine cache reusable. Still PNGs are cached similarly by program,
dimensions, step count, binning and palette version. Temporary files are
renamed only after successful rendering. Delete a specific cache entry to
force its regeneration; bump the version suffix when changing render semantics.

The 4K script verifies the encoded dimensions, frame count, frame rate and
duration with `ffprobe` and writes a JSON manifest including the fixed layout
and time schedule. It also saves first, midpoint and final PNGs.

After a full frame cache exists, smoke-check first/midpoint/final composition
without encoding another one-minute video or overwriting the original output:

```bash
python experiments/bb6-holdouts/holdout_grid_4k.py /tmp/bb6-output --preview --output-dir /tmp/bb6-smoke
python experiments/bb6-holdouts/holdout_landscape.py /tmp/bb6-output --output-dir /tmp/bb6-smoke
python experiments/bb6-holdouts/holdout_title.py /tmp/bb6-smoke
```

`--preview` encodes three frames at 30 fps (0.1 seconds), sampling the same
simulation targets as the full movie. It validates that shorter duration and
uses a separate filename. If a cache is missing, it is generated first.

## Optional appearance analysis and fast montage

The 60.8-second montage shows machines in original list order, three snapshots
per machine (3,333, 6,666 and 10,000 steps) at 30 fps. Its renderer also supplies
the high-resolution 10K inputs used by the appearance experiment.

```bash
python experiments/bb6-holdouts/holdout_montage.py /tmp/bb6-output
python experiments/bb6-holdouts/holdout_appearance.py /tmp/bb6-output
python experiments/bb6-holdouts/holdout_snake_oriented.py /tmp/bb6-output
```

Appearance comparison uses foreground intensity, normalized equally weighted
16 × 16 and 64 × 64 images, and the minimum cosine distance under horizontal
reflection. Average-linkage clustering plus optimal leaf ordering gives the
linear snake sequence. The 2D comparisons use classical MDS, assignment to
grid slots and 150,000 greedy swaps with random seed 600. The final-row-only
comparison uses the bottom row of the binned diagram, not the exact final tape.

These commands save new candidate layouts under `CACHE_ROOT/appearance/`.
They never replace the checked-in published layout. Numerical-library changes
can change a recomputed ordering; use the frozen files for the published result.

## Explicit Rust entry points

No example runs during normal library or website use. Python invokes these
native examples explicitly; they use existing development dependencies:

```bash
cargo run --offline --release --example holdout_grid_frames -- experiments/bb6-holdouts/data/bb6_holdouts.txt /tmp/bb6-output 100000 118 80
cargo run --offline --release --example holdout_stills -- experiments/bb6-holdouts/data/bb6_holdouts.txt /tmp/bb6-output 100000 236 184
cargo run --offline --release --example holdout_frames -- experiments/bb6-holdouts/data/bb6_holdouts.txt /tmp/bb6-output
```

The Python wrappers use Cargo offline mode. Before the first offline run on
a new checkout, run `cargo fetch --locked` with network access. The machine
list and layouts themselves are local, frozen inputs and require no fetching.
