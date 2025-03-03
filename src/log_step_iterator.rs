/// A logarithmic iterator that generates `num_frames` steps between 0 and `max_value`, inclusive.
pub struct LogStepIterator {
    current_frame: u32,
    total_frames: u32,
    max_value: u64,
}

impl LogStepIterator {
    #[inline]
    #[must_use]
    pub const fn new(max_value: u64, total_frames: u32) -> Self {
        Self {
            current_frame: 0,
            total_frames,
            max_value,
        }
    }
}

#[allow(
    clippy::missing_trait_methods,
    clippy::min_ident_chars,
    clippy::float_cmp
)]
impl Iterator for LogStepIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_frame >= self.total_frames {
            return None;
        }

        // Normalized frame index from 0 to 1
        let t = self.current_frame as f64 / (self.total_frames - 1) as f64;

        // Apply logarithmic-like spacing using an exponential function
        let value = if t == 0.0 {
            0
        } else if t == 1.0 {
            self.max_value - 1
        } else {
            let log_value = ((self.max_value as f64).ln() * t).exp();
            log_value.round().min((self.max_value - 1) as f64) as u64
        };

        self.current_frame += 1;
        Some(value)
    }
}
