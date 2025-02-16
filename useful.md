# Useful Commands

## Run Fastest

```bash
set RUSTFLAGS=-C target-cpu=native -C opt-level=3
cargo build --release
```

## Rust image test

```bash
cargo test bb5_champ_space_time --release -- --nocapture
```

## Package for WASM

```bash
wasm-pack build --out-dir docs/v0.2.2/pkg --target web
```

## Make movie frames

```bash
cargo run --example movie --release bb5_champ 4K
```

## Tetration

```bash
 cargo run --example tetration --release
 cargo test --example tetration --release -- --nocapture
```
