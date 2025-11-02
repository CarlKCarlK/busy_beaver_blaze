#![allow(clippy::float_arithmetic)]

use std::str::FromStr;
use std::time::Instant;

use busy_beaver_blaze::{BB5_CHAMP, CompiledFnId, Config, Machine};
use thousands::Separable;

const RUNS: usize = 50;
const MIN_TAPE: usize = 2_097_152;
const MAX_TAPE: usize = 16_777_216;
const ASM_INTERVAL: u64 = 50_000_000;

fn run_rust_machine() -> (u64, u32) {
    let mut machine = Machine::from_str(BB5_CHAMP).expect("valid BB5_CHAMP");
    let mut steps = 0_u64;

    loop {
        let result = machine.next();
        steps += 1;
        if result.is_none() {
            break;
        }
    }

    let ones = machine.count_nonblanks();
    (steps, ones)
}

fn run_asm_machine() -> (u64, u32) {
    let config = Config::new(
        CompiledFnId::Bb5Champ,
        ASM_INTERVAL,
        u64::MAX,
        MIN_TAPE,
        MAX_TAPE,
    )
    .expect("config")
    .with_quiet(true);

    let summary = config.run();
    let ones = summary.tape().iter().filter(|&&symbol| symbol != 0).count();
    let ones = u32::try_from(ones).expect("nonblank count fits in u32");

    (summary.step_count, ones)
}

fn main() {
    println!(
        "Benchmarking BB5_CHAMP halting runs with {} Rust and {} asm executions",
        RUNS, RUNS
    );

    let rust_start = Instant::now();
    let mut rust_result = (0_u64, 0_u32);
    for _ in 0..RUNS {
        rust_result = run_rust_machine();
    }
    let rust_total = rust_start.elapsed().as_secs_f64();
    let rust_avg = rust_total / RUNS as f64;

    let asm_start = Instant::now();
    let mut asm_result = (0_u64, 0_u32);
    for _ in 0..RUNS {
        asm_result = run_asm_machine();
    }
    let asm_total = asm_start.elapsed().as_secs_f64();
    let asm_avg = asm_total / RUNS as f64;

    assert_eq!(rust_result, asm_result, "results must match");

    println!(
        "Rust interpreter: {:.3}s per run (total {:.3}s)",
        rust_avg, rust_total
    );
    println!(
        "Inline asm runner: {:.3}s per run (total {:.3}s)",
        asm_avg, asm_total
    );
    println!(
        "Speedup: {:.2}x (steps {}, ones {})",
        rust_avg / asm_avg,
        rust_result.0.separate_with_commas(),
        rust_result.1.separate_with_commas()
    );
}
