# Useful Commands

## Python Dev mode

```bash
uv pip install -e .
uv pip install -e ".[dev]"
python -m pytest
```

## Run Fastest

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
wasm-pack build --release --out-dir docs/v0.2.5/pkg --target web && del docs\v0.2.5\pkg\.gitignore
wasm-opt -Oz --strip-debug --strip-dwarf -o docs/v0.2.5/pkg/busy_beaver_blaze_bg.wasm docs/v0.2.5/pkg/busy_beaver_blaze_bg.wasm
```

### Debug

```bash
wasm-pack build --out-dir docs/v0.2.5/pkg --target web && del docs\v0.2.5\pkg\.gitignore
```

## Make movie frames

```bash
cargo run --example movie --release bb5_champ 2K true
cargo run --example movie --release bb6_contender 2K true
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
