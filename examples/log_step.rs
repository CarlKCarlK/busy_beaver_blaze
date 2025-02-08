use busy_beaver_blaze::LogStepIterator;

fn main() {
    let max_value = 1_000_000_000_000; // 1 trillion
    let num_frames = 10;

    let log_iter = LogStepIterator::new(max_value, num_frames);

    for (index, step) in log_iter.enumerate() {
        println!("Frame {}: {}", index, step);
    }
}
