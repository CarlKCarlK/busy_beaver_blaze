# Useful Commands

## Testing

**Rust Tests:**

```bash
cargo check --all-features --all-targets
cargo test --all-features --all-targets --release
```

**Python Tests:**

```bash
maturin develop --release --features python
pytest
```

## Run Turing Compiler

```bash
# Can also use defaults
cargo run --example compile_machine --release -- --program bb6-contender --interval 1_000_000_000 --max-steps 25,000,000,000

# expand the assembly code (then ask AI to annotate it)
cargo expand --example compile_machine bb6_contender_compiled
```

## Python Dev mode (uv workflow)

**Python Initial setup (one time):**

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
```

## Package for WASM

**WASM target (required for web builds):**

```bash
rustup target add wasm32-unknown-unknown
```

### Release (ES6 modules - for GitHub Pages with HTTPS)

```bash
#cargo install wasm-opt --locked
wasm-pack build --release --out-dir docs/v0.2.7/pkg --target web && del docs\v0.2.7\pkg\.gitignore
wasm-opt -Oz --strip-debug --strip-dwarf -o docs/v0.2.7/pkg/busy_beaver_blaze_bg.wasm docs/v0.2.7/pkg/busy_beaver_blaze_bg.wasm
```

### Release (no-modules - for Pico W http:// non-secure context)

```bash
wasm-pack build --release --out-dir docs/v0.2.7/pkg --target no-modules && del docs\v0.2.7\pkg\.gitignore
wasm-opt -Oz --strip-debug --strip-dwarf -o docs/v0.2.7/pkg/busy_beaver_blaze_bg.wasm docs/v0.2.7/pkg/busy_beaver_blaze_bg.wasm
```

### Debug

```bash
wasm-pack build --out-dir docs/v0.2.7/pkg --target web && del docs\v0.2.7\pkg\.gitignore
```

## Tetration

```bash
 cargo run --example tetration --release
 ``

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
