# Useful Commands

## Run Turing Compiler

```bash
cargo run --example compile_machine --release
cargo test --example compile_machine
```

## Python Dev mode

```bash
uv pip install -e .
uv pip install -e ".[dev]"
python -m pytest
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
wasm-pack build --release --out-dir docs/v0.2.6/pkg --target web && del docs\v0.2.6\pkg\.gitignore
wasm-opt -Oz --strip-debug --strip-dwarf -o docs/v0.2.6/pkg/busy_beaver_blaze_bg.wasm docs/v0.2.6/pkg/busy_beaver_blaze_bg.wasm
```

### Debug

```bash
wasm-pack build --out-dir docs/v0.2.6/pkg --target web && del docs\v0.2.6\pkg\.gitignore
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
