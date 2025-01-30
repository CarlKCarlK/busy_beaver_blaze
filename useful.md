# Useful Commands

## Run Fastest

```bash
set RUSTFLAGS=-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1 -C panic=abort -C embed-bitcode=yes
cargo build --release
```
