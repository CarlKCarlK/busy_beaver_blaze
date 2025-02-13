# Busy Beaver Blaze

This repository contains a Turing machine interpreter and space time visualization implemented in Rust and compiled to WebAssembly.

* [Run this program](https://carlkcarlk.github.io/busy_beaver_blaze/) in your own browser.
* Run the champion [Busy Beaver](https://en.wikipedia.org/wiki/Busy_beaver) Turing machines for millions of steps in less than a second.
* Run your own Turing machines.
* Visualize Turing machines with space time diagrams.
* Visualize of millions of steps in less than a second. Visualize of a billion steps in about 5 seconds.

## Techniques

* The Turing machine interpreter is a straight forward implementation in Rust.
* The space-time visualization is implemented via adaptive sampling. The sampler starts by recording the whole tape
at every machine step. If the tape or steps grows beyond
twice the size of the desired image, the sampler reduces the sampling rate by half. Total memory and time used is, thus, proportional to the size of the desired image, not the number of machine steps or width of tape visited.
* The port to WASM followed [Nine Rules for Running Rust in the Browser](https://medium.com/towards-data-science/nine-rules-for-running-rust-in-the-browser-8228353649d1) in *Towards Data Science*.

## License

This project is dual-licensed under either:

* MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)

at your option.

## Contributing

Contributing to this project is welcome. Feature and bug reports are appreciated.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project shall be dual-licensed as above, without any additional terms or conditions.
