# Useful Commands

<!-- cmk update these -->

## Testing

**Rust Tests:**

```bash
cargo test --features python
```

**Python Tests:**

```bash
maturin develop --release --features python
pytest tests/python/
```

**Integration Test:**

```bash
python quick_demo.py
python examples/movie_list.py
```

### Build & Install

**Development Mode:**

```bash
# Set Python interpreter
$env:PYO3_PYTHON = "py"

# Install in editable mode
maturin develop --release --features python

# Or install with pip
pip install -e ".[dev]"
```

**Production Build:**

```bash
maturin build --release --features python
# Wheel created in target/wheels/
```

### Usage Example

```python
from busy_beaver_blaze import (
    PngDataIterator,
    BB5_CHAMP,
    log_step_iterator,
    create_frame,
    RESOLUTION_2K,
)

# Generate frame indices
frame_steps = log_step_iterator(1_000_000, 100)

# Create iterator (multithreaded in Rust)
iterator = PngDataIterator(
    early_stop=1_000_000,
    program=BB5_CHAMP,
    width=RESOLUTION_2K[0],
    height=RESOLUTION_2K[1],
    pixel_policy="binning",
    frame_steps=frame_steps,
    colors=[],  # Empty = use defaults
    part_count=0  # 0 = auto-detect CPU count
)

# Process frames one-at-a-time (memory efficient)
for step_index, png_bytes in iterator:
    # Add text overlay and resize (Python/PIL)
    frame = create_frame(png_bytes, "BB5", step_index, 1920, 1080)
    frame.save(f"frame_{step_index:07d}.png")
```


## Run Turing Compiler

```bash
cargo test --example compile_machine
cargo run --example compile_machine --release -- --max-steps 100_000_000
cargo run --example compile_machine --release -- --program bb6-contender --interval 100_000_000_000  --min-tape 4 --max-tape 1,000,000,000,000,000
cargo run --example compile_machine --release -- --program bb6-contender --interval 1_000_000_000 --max-steps 25,000,000,000

# expand the assembly code (then ask AI to annotate it)
cargo expand --example compile_machine bb6_contender_compiled
```

## Python Dev mode (uv workflow)

**Initial setup (one time):**

```bash
# Create venv and install all dependencies including dev extras
uv sync --extra dev

# Build and install the Rust extension
uv run maturin develop --release --features python
```

**Activate the virtual environment:**

```bash
# PowerShell
.\.venv\Scripts\Activate.ps1

# CMD
.venv\Scripts\activate.bat

# Git Bash / WSL
source .venv/Scripts/activate

# Or use uv run to run commands without activating
uv run pytest
uv run python quick_demo.py
```

**After activating, you can run commands directly:**

```bash
pytest tests/python/
maturin develop --release --features python
python examples/movie_list.py
```

**Quick rebuild after code changes:**

```bash
uv run maturin develop --release --features python
```

## Run Fastest

Set up

```bash
rustup toolchain install nightly
rustup target add wasm32-unknown-unknown --toolchain nightly
rustup override set nightly
```

```bash
set RUSTFLAGS=-C target-cpu=native -C opt-level=3
cargo build --release
```

## Rust image test

```bash
cargo test bb5_champ_space_by_time --release -- --nocapture
```

## Package for WASM

### Release

```bash
#cargo install wasm-opt --locked
wasm-pack build --release --out-dir docs/v0.2.7/pkg --target web && del docs\v0.2.7\pkg\.gitignore
wasm-opt -Oz --strip-debug --strip-dwarf -o docs/v0.2.7/pkg/busy_beaver_blaze_bg.wasm docs/v0.2.7/pkg/busy_beaver_blaze_bg.wasm
```

### Debug

```bash
wasm-pack build --out-dir docs/v0.2.7/pkg --target web && del docs\v0.2.7\pkg\.gitignore
```

## Make movie frames

```bash
sudo run PowerShell -Command "Stop-Service WSearch"
cargo run --example movie --release bb5_champ 2K true
cargo run --example movie --release bb6_contender 2K true
cargo run --example movie --release BB_3_3_355317 2K true
cargo run --example movie_list --release
```

## Tetration

```bash
 cargo run --example tetration --release
 cargo test --example tetration --release -- --nocapture
```

## testing

```bash
cargo run --example movie --release bb5_champ tiny
start cargo run --example frame --release bb6_contender 2k 14 14 0 
cargo test benchmark1 --release -- --nocapture
cargo test benchmark2 --release --target wasm32-unknown-unknown
cargo test benchmark2 --release -- --nocapture
cargo test benchmark63 --release -- --nocapture
```

## Criterion Benchmark

```bash
cargo bench
```
