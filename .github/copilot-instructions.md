# Busy Beaver Blaze - AI Development Guide

## Architecture Overview

This is a high-performance Turing machine interpreter and space-time visualizer with **dual implementations** (Python + Rust) and **WebAssembly compilation** for browser deployment.

### Core Components

- **`Machine`** (`src/machine.rs`): Core Turing machine interpreter with state/tape/program
- **`Tape`** (`src/tape.rs`): Infinite tape using two `AVec<Symbol>` (negative/nonnegative indices)
- **`Spaceline`** (`src/spaceline.rs`): Horizontal slice of tape at one time step, with SIMD-optimized pixel compression
- **`Spacelines`** (`src/spacelines.rs`): Collection of spacelines with Y-axis compression buffering
- **`SpaceByTime`** (`src/space_by_time.rs`): Full space-time diagram manager with adaptive sampling
- **`SpaceByTimeMachine`** (`src/space_by_time_machine.rs`): WebAssembly-exposed API combining machine + visualization

### Performance Architecture

**Adaptive Sampling**: Memory scales with image size, NOT step count or tape width
- Starts recording full tape at each step
- If tape/steps exceed 2x image size → halves sampling rate
- Uses `PowerOfTwo` types for stride calculations (`src/power_of_two.rs`)

**SIMD Optimization**: Controlled by `simd` feature flag
- Pixel binning uses SIMD lanes (8, 16, 32, 64) for averaging tape symbols
- Falls back to iterator-based implementations when SIMD unavailable
- Memory alignment via `ALIGN: usize = 64` constant

## Variable Naming Conventions

**Avoid single-character variables** - use descriptive names:
- ❌ `i`, `j`, `x`, `y`, `a`, `b` 
- ✅ `read_index`, `write_index`, `first_pixel`, `second_pixel`

**Project patterns**:
- `x_goal`/`y_goal`: Target image dimensions 
- `x_stride`/`y_stride`: Sampling rates (must be `PowerOfTwo`)
- `step_index`: Current machine step number
- `tape_index`: Current head position (can be negative)
- `select`: Which symbol to visualize (`NonZeroU8`)

## Comment Conventions

Use `cmk00`/`cmk0` prefix for TODO items (author's initials + priority):
```rust
// cmk00 high priority task
// cmk0 lower priority consideration
// TODO standard todo for general items
```

## Build & Test Workflows

### Development Commands
```bash
# Rust native (fastest)
cargo test --release -- --nocapture
cargo run --example movie --release bb5_champ 2K true

# WebAssembly build
wasm-pack build --release --out-dir docs/v0.2.6/pkg --target web
wasm-opt -Oz --strip-debug --strip-dwarf -o docs/v0.2.6/pkg/busy_beaver_blaze_bg.wasm docs/v0.2.6/pkg/busy_beaver_blaze_bg.wasm

# Python development
uv pip install -e ".[dev]"
python -m pytest
```

### Key Test Patterns
- Image comparison tests in `tests/expected/*.png`
- SIMD vs non-SIMD equivalence testing
- Cross-platform WASM target testing: `cargo test --target wasm32-unknown-unknown`

## Common Pitfalls

**Memory Alignment**: Use `AVec` with `ALIGN` constant, not `Vec`
```rust
let mut pixels: AVec<Pixel> = AVec::with_capacity(ALIGN, capacity);
```

**PowerOfTwo Arithmetic**: Use dedicated methods, not raw operations
```rust
let stride = PowerOfTwo::from_exp(3); // Good
let count = stride.div_ceil_into(values.len()); // Good
let bad = values.len() / stride.as_usize(); // May truncate
```

**Feature Flag Patterns**: Always provide non-SIMD fallbacks
```rust
#[cfg(feature = "simd")]
let result = simd_function();
#[cfg(not(feature = "simd"))]
let result = iterator_function();
```

**WASM Bindings**: Use `#[wasm_bindgen]` on public APIs, handle JS bigint conversions
```rust
#[wasm_bindgen]
pub fn step_count(&self) -> u64 { /* return step count */ }
```

## Integration Points

- **Web Worker**: `docs/*/worker.js` handles heavy computation off main thread
- **Three Parsing Formats**: Symbol Major, State Major, BB Challenge Standard Format
- **Color Customization**: `normalize_colors()` cycles through palettes for multi-symbol visualization
- **URL State**: Web app supports hash fragments like `#program=bb5&earlyStop=false&run=true`

Reference `src/code_notes.md` for sampling vs binning implementation locations across the codebase.