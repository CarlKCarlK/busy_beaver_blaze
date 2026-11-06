use crate::{PixelPolicy, PyramidMachine, normalize_colors};

/// Sequential Square Root time-scale frames, using the web renderer's checkpoints.
///
/// Frame steps are zero-based visualization indices: zero is the initial blank
/// tape. Iteration stops at the last requested index or when the machine halts.
pub struct SqrtPngDataIterator {
    machine: PyramidMachine,
    frame_steps: std::vec::IntoIter<u64>,
    colors: Vec<u8>,
}

impl SqrtPngDataIterator {
    /// Create a lazy frame iterator with no worker threads or time partitioning.
    ///
    /// # Errors
    /// Returns an error for empty/decreasing frame steps, invalid image dimensions,
    /// an invalid program, or an invalid palette.
    pub fn new(
        frame_steps: Vec<u64>,
        program: &str,
        (x_goal, y_goal): (u32, u32),
        pixel_policy: PixelPolicy,
        colors: &[[u8; 3]],
    ) -> Result<Self, String> {
        if frame_steps.is_empty() {
            return Err("frame_steps cannot be empty".into());
        }
        if !frame_steps.windows(2).all(|steps| steps[0] <= steps[1]) {
            return Err("frame_steps must be monotonically increasing".into());
        }
        // Rendering normalizes this palette to the program's symbol count.
        let palette_len = if colors.is_empty() {
            crate::DEFAULT_COLORS.len()
        } else {
            colors.len()
        };
        let colors = normalize_colors(colors, palette_len).map_err(|error| error.to_string())?;
        let machine = match pixel_policy {
            PixelPolicy::Binning => PyramidMachine::new(program, x_goal, y_goal, "sqrt"),
            PixelPolicy::Sampling => PyramidMachine::sampling(program, x_goal, y_goal, "sqrt"),
        }?;
        Ok(Self {
            machine,
            frame_steps: frame_steps.into_iter(),
            colors: colors.into_iter().flatten().collect(),
        })
    }
}

impl Iterator for SqrtPngDataIterator {
    type Item = Result<(u64, Vec<u8>), String>;

    fn next(&mut self) -> Option<Self::Item> {
        let step_index = self.frame_steps.next()?;
        while self.machine.step_count() - 1 < step_index {
            if !self.machine.step() {
                // Never label a halted image with a requested but unreached step.
                self.frame_steps = Vec::new().into_iter();
                return None;
            }
        }
        Some(
            self.machine
                .to_png(&self.colors)
                .map(|png| (step_index, png)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_match_web_renderer_across_checkpoints_and_compression() {
        let colors = [[255, 255, 255], [255, 0, 0], [0, 0, 255]];
        let flat_colors: Vec<u8> = colors.into_iter().flatten().collect();
        for policy in [PixelPolicy::Binning, PixelPolicy::Sampling] {
            let program = crate::BB_3_3_355317;
            let steps = vec![0, 1, 1, 3, 8, 17, 64, 129];
            let frames: Vec<_> =
                SqrtPngDataIterator::new(steps.clone(), program, (8, 4), policy, &colors)
                    .unwrap()
                    .collect();
            assert_eq!(frames.len(), steps.len());
            let mut reference = match policy {
                PixelPolicy::Binning => PyramidMachine::new(program, 8, 4, "sqrt"),
                PixelPolicy::Sampling => PyramidMachine::sampling(program, 8, 4, "sqrt"),
            }
            .unwrap();
            for (expected_step, frame) in steps.into_iter().zip(frames) {
                while reference.step_count() - 1 < expected_step {
                    assert!(reference.step());
                }
                let (step_index, png) = frame.unwrap();
                assert_eq!(step_index, expected_step);
                assert_eq!(png, reference.to_png(&flat_colors).unwrap());
            }
        }
    }

    #[test]
    fn halt_ends_iteration_without_unreached_frames() {
        let mut frames =
            SqrtPngDataIterator::new(vec![0, 1, 100], "1RZ1RZ", (8, 4), PixelPolicy::Binning, &[])
                .unwrap();
        assert_eq!(frames.next().unwrap().unwrap().0, 0);
        assert!(frames.next().is_none());
        assert!(frames.next().is_none());
    }
}
