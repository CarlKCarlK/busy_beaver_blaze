//! Repeatable native baseline; includes stepping and periodic PNG rendering.
use busy_beaver_blaze::{BB5_CHAMP, SpaceByTimeMachine};
use std::{hint::black_box, time::Instant};
fn main() {
    for binning in [true, false] {
        for _trial in 0..5 {
            let mut machine =
                SpaceByTimeMachine::from_str(BB5_CHAMP, 128, 128, binning, 0).unwrap();
            let start = Instant::now();
            for _frame in 0..10 {
                assert!(machine.nth_js(99_999));
                black_box(machine.to_png(&[255, 255, 255, 255, 165, 0]).unwrap());
            }
            println!(
                "binning={binning} elapsed_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
}
