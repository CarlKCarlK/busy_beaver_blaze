# Useful Commands

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

```bash
wasm-pack build --out-dir docs/v0.2.4/pkg --target web && del docs\v0.2.4\pkg\.gitignore
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
