# Busy Beaver Implementation in Rust

This repository contains a Rust implementation of the [Busy Beaver](https://en.wikipedia.org/wiki/Busy_beaver) Turing machine simulator.

## Overview

The program simulates a Turing machine with the following characteristics:

- *N* states
- 2 symbols (0 and 1)
- A bidirectional infinite tape
- Debugging output at configurable intervals

## Usage

```bash
cargo run --release
```

```rust
fn main() -> Result<(), Error> {
    const STATE_COUNT: usize = 5;
    let program: Program<STATE_COUNT> = BB5_CHAMP.parse()?;

    let mut machine: Machine<'_, STATE_COUNT> = Machine {
        tape: Tape::default(),
        tape_index: 0,
        program: &program,
        state: 0,
    };

    let debug_interval = 10_000_000;
    let step_count = machine.debug_count(debug_interval);

    println!(
        "Final: Step {}: {:?}, #1's {}",
        step_count.separate_with_commas(),
        machine,
        machine.tape.count_ones()
    );

    Ok(())
}
```

Outputs:

```text
Step 0: Machine { state: 1, tape_index: 1}
Step 10,000,000: Machine { state: 3, tape_index: -2351}
Step 20,000,000: Machine { state: 1, tape_index: -5031}
Step 30,000,000: Machine { state: 1, tape_index: -4427}
Step 40,000,000: Machine { state: 1, tape_index: -6559}
Final: Step 47,176,869: Machine { state: 7, tape_index: -12242}, #1's 4098
```

## Wishlist

- WASM version
- Visualizations
- Timing
- Testing, especially for parsing
- See if tape could be more efficient with bit arrays

## License

This project is dual-licensed under either:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)

at your option.

## Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project shall be dual-licensed as above, without any additional terms or conditions.
