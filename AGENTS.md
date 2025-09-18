# Busy Beaver Blaze - AI Development Guide

## Architecture Overview

This is a high-performance Turing machine interpreter and space-time visualizer with **dual implementations** (Python + Rust) and **WebAssembly compilation** for browser deployment.

### Core Components

- `Machine` (`src/machine.rs`): Core Turing machine interpreter with state/tape/program
- `Tape` (`src/tape.rs`): Infinite tape using two `AVec<Symbol>` (negative/nonnegative indices)
- `Spaceline` (`src/spaceline.rs`): Horizontal slice of tape at one time step, with SIMD-optimized pixel compression
- `Spacelines` (`src/spacelines.rs`): Collection of spacelines with Y-axis compression buffering
- `SpaceByTime` (`src/space_by_time.rs`): Full space-time diagram manager with adaptive sampling
- `SpaceByTimeMachine` (`src/space_by_time_machine.rs`): WebAssembly-exposed API combining machine + visualization

### Performance Architecture

**Adaptive Sampling**: Memory scales with image size, NOT step count or tape width

- Starts recording full tape at each step
- If tape/steps exceed 2x image size → halves sampling rate
- Uses `PowerOfTwo` types for stride calculations (`src/power_of_two.rs`)

**SIMD Optimization**: Controlled by `simd` feature flag

- Pixel binning uses SIMD lanes (8, 16, 32, 64) for averaging tape symbols
- Falls back to iterator-based implementations when SIMD unavailable
- Memory alignment via `ALIGN: usize = 64` constant

**Diff Row Optimization**: Controlled by `diff_row` feature flag (default ON)

- Optimizes frame generation by copying the previous spaceline and recomputing only the single pixel that can change per machine step.
- Gate is in `src/space_by_time.rs: snapshot()`; recompute-from-scratch fallback is used when `--no-default-features` or when `diff_row` is disabled.
- Pixel recompute is implemented in `src/spaceline.rs: redo_pixel()`.
- Build examples:
  - Default (SIMD + Diff Row): `cargo check`
  - Disable both: `cargo check --no-default-features`
  - SIMD only: `cargo check --no-default-features --features simd`
  - Diff Row only: `cargo check --no-default-features --features diff_row`

## Variable Naming Conventions

Avoid single-character variables; use descriptive names:

- ❌ `i`, `j`, `x`, `y`, `a`, `b`
- ✅ `read_index`, `write_index`, `first_pixel`, `second_pixel`

Project patterns:

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

Preserving comments: When changing code, generally don't remove TODO's and cmk's in comments. Just move the comments if needed. If you think they no longer apply, add `(may no longer apply)` to the comment rather than deleting it.

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

- Web Worker: `docs/*/worker.js` handles heavy computation off main thread
- Three Parsing Formats: Symbol Major, State Major, BB Challenge Standard Format
- Color Customization: `normalize_colors()` cycles through palettes for multi-symbol visualization
- URL State: Web app supports hash fragments like `#program=bb5&earlyStop=false&run=true`

Reference `src/code_notes.md` for sampling vs binning implementation locations across the codebase.

## AI Agent Guidelines

- Action-first workflow: read relevant modules before editing (`machine.rs`, `spaceline.rs`, `spacelines.rs`, `space_by_time.rs`, `space_time_layers.rs`). Prefer surgical diffs over large refactors.
- Preserve comments: keep `cmk00`/`cmk0`/`TODO` comments. If they seem obsolete, append `(may no longer apply)` rather than deleting.
- Feature flags: always provide a clear fallback path. Use `#[cfg(feature = "..."))]` and `#[cfg(not(feature = "..."))]` blocks with identical control flow where possible.
- New optimizations: hide behind a feature flag defaulting appropriately. Update this guide and `Cargo.toml` features list.
- Naming/style: avoid single-letter names; use project patterns (`x_goal`, `y_stride`, `step_index`, `tape_index`, `select`). Use `PowerOfTwo` helpers instead of raw shifts/divs.
- Memory/perf: use `AVec` with `ALIGN` for hot paths; avoid extra allocations and unnecessary `clone()`; prefer slice views and in-place ops.
- SIMD usage: guard with `simd` feature; keep safe fallbacks. Document any `unsafe` with a clear safety comment and alignment guarantees.
- WASM constraints: avoid OS/thread-only APIs in shared code. Keep public APIs annotated with `#[wasm_bindgen]` where they are exposed.
- Test matrix: run checks across features before submitting changes:
  - `cargo check`
  - `cargo check --no-default-features`
  - `cargo check --no-default-features --features simd`
  - `cargo check --no-default-features --features diff_row`
  - `cargo test --release` (native), and optionally `cargo test --target wasm32-unknown-unknown`
- Benchmarks: validate perf-sensitive changes with `cargo bench` and compare to prior runs; ensure functional equivalence across feature variants.
- Public API stability: avoid renaming exported types/functions without coordination. Add doc comments for any new public items.
- Documentation: when adding a feature or toggling behavior, update this guide and any relevant docs/examples.

### Precision Over Future‑Proofing

- Prefer precise code that encodes current assumptions with `assert!`s and fails fast when violated.
- Do not write code that is “resilient to possible future changes” at the expense of clarity; instead, express today’s preconditions explicitly and let assertions catch regressions if behavior changes later.
- When control flow has expected invariants (e.g., counters must be equal, ranges nonnegative), use `assert!` rather than saturating math or silent fallbacks.
- If a `match` requires a catch‑all only for type completeness, use `unreachable!()`/`panic!()` rather than `_ => {}` to surface violations early.

- When you write assembly code (for example using `asm!` in Rust) include a comment explaining what that statement does.

- In Rust, generally custom Error enums should be named 'Error' rather than 'MyThingError'

- In Rust, move deconstruction into the arguments were possible.

- In Rust, I like using the same name when unwrapping, if let Some(max_steps) = max_steps {

- I like asserts and using asserts. So, if the difference between two values must always be nonnegative, I would NOT use saturating_sub, I would use assert!(a >= b); let diff = a - b; because I want to catch any violations. Likewise, if a match requires a catch all, I wold use unreachable or panic. I would not use_ => {}.
