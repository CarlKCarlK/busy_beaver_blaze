//! Measure pyramid capture, reconstruction, PNG encoding and retained layer storage.
use busy_beaver_blaze::{BB5_CHAMP, PyramidMachine, SpaceByTimeMachine};
use std::{hint::black_box, time::Instant};
fn main() {
    for (name, program) in [
        ("bb5", BB5_CHAMP.to_owned()),
        (
            "26 layers",
            (0..26)
                .map(|state_index| {
                    let symbol = char::from_digit(state_index + 1, 36).unwrap();
                    let next_state = char::from(b'A' + ((state_index + 1) % 26) as u8);
                    format!("{symbol}R{next_state}").repeat(27)
                })
                .collect::<Vec<_>>()
                .join("_"),
        ),
    ] {
        for (width, height) in [(128, 128), (360, 432)] {
            let mut baseline =
                SpaceByTimeMachine::from_str(&program, width, height, true, 0).unwrap();
            let baseline_start = Instant::now();
            assert!(baseline.nth_js(999_999));
            black_box(baseline.step_count());
            let baseline_ms = baseline_start.elapsed().as_secs_f64() * 1000.0;
            drop(baseline);
            println!("{name} {width}x{height} baseline_capture_ms={baseline_ms:.3}");
            let mut machine = PyramidMachine::new(&program, width, height, "linear-b").unwrap();
            let start = Instant::now();
            for _step_index in 0..1_000_000 {
                assert!(machine.step());
            }
            let capture_ms = start.elapsed().as_secs_f64() * 1000.0;
            println!(
                "{name} {width}x{height} capture_ms={capture_ms:.3} checkpoints={} retained_bytes={}",
                machine.checkpoint_count(),
                machine.retained_bytes()
            );
            for scale in ["linear-b", "sqrt", "log"] {
                machine.set_time_scale(scale).unwrap();
                let start = Instant::now();
                for _frame in 0..5 {
                    black_box(machine.to_png(&[255, 255, 255, 255, 165, 0]).unwrap());
                }
                println!(
                    "  {scale} render_png_ms={:.3}",
                    start.elapsed().as_secs_f64() * 200.0
                );
            }
        }
    }
}
