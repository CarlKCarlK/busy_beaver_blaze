use crate::{
    Error, Machine, Pixel, encode_png, power_of_two::PowerOfTwo, sample_rate, spaceline::Spaceline,
    spacelines::Spacelines,
};

pub struct SpaceByTime {
    pub(crate) step_index: u64,
    x_goal: u32,
    y_goal: u32,
    sample: PowerOfTwo,
    spacelines: Spacelines,
    x_smoothness: PowerOfTwo,
    y_smoothness: PowerOfTwo,
    previous_space_line: Option<Spaceline>,
}

/// Create a new in which you give the `x_goal` (space)
/// and the `y_goal` (time). The sample starts at 1 and
/// inner is a vector of one spaceline
impl SpaceByTime {
    #[inline]
    #[must_use]
    pub fn new(
        x_goal: u32,
        y_goal: u32,
        x_smoothness: PowerOfTwo,
        y_smoothness: PowerOfTwo,
        buffer1_count: PowerOfTwo,
    ) -> Self {
        Self {
            step_index: 0,
            x_goal,
            y_goal,
            sample: PowerOfTwo::ONE,
            spacelines: Spacelines::new(x_smoothness, buffer1_count),
            x_smoothness,
            y_smoothness,
            previous_space_line: None,
        }
    }

    fn compress_if_needed(&mut self) {
        // Sampling & Averaging 1--
        // We sometimes need to squeeze rows by averaging adjacent pairs of rows.
        // The alternative is just to keep the 1st row and discard the 2nd row.

        let new_sample = sample_rate(self.step_index, self.y_goal);
        if new_sample != self.sample {
            assert!(
                new_sample / self.sample == PowerOfTwo::TWO,
                "real assert 10"
            );
            self.sample = new_sample;
            if new_sample <= self.y_smoothness {
                self.spacelines.compress_average();
            } else {
                self.spacelines.compress_take_first(new_sample);
            }
        }
    }

    // ideas
    // use
    //       assert!(self.sample.is_power_of_two(), "Sample must be a power of two");
    //       // Use bitwise AND for fast divisibility check
    //       if self.step_index & (self.sample - 1) != 0 {
    //  Also: Inline the top part of the function.
    //  Maybe pre-subtract 1 from sample

    #[inline]
    pub(crate) fn snapshot(&mut self, machine: &Machine, previous_tape_index: i64) {
        self.step_index += 1;
        let inside_index = self.sample.rem_into_u64(self.step_index);

        if inside_index == 0 {
            // We're starting a new set of spacelines, so flush the buffer and compress (if needed)
            self.spacelines.flush_buffer0();
            self.compress_if_needed();
        }

        let down_step = self.sample.saturating_div(self.y_smoothness);
        if !down_step.divides_u64(inside_index) {
            return;
        }
        let inside_inside_index = down_step.divide_into(inside_index);
        let down_step_one = down_step == PowerOfTwo::ONE;

        let spaceline =
            if let Some(mut previous) = self.previous_space_line.take().filter(|_| down_step_one) {
                // cmk messy code
                if previous.redo_pixel(
                    previous_tape_index,
                    &machine.tape,
                    self.x_goal,
                    self.step_index,
                    self.x_smoothness,
                ) {
                    previous
                } else {
                    Spaceline::new(
                        &machine.tape,
                        self.x_goal,
                        self.step_index,
                        self.x_smoothness,
                    )
                }
            } else {
                Spaceline::new(
                    &machine.tape,
                    self.x_goal,
                    self.step_index,
                    self.x_smoothness,
                )
            };
        if down_step_one {
            self.previous_space_line = Some(spaceline.clone());
        }

        self.spacelines
            .push(inside_inside_index, self.sample, spaceline);
    }

    #[allow(clippy::wrong_self_convention)] // cmk00 consider better name to this function
    pub fn to_png(&mut self) -> Result<Vec<u8>, Error> {
        let last = self
            .spacelines
            .last(self.step_index, self.sample, self.y_smoothness);
        let x_sample: PowerOfTwo = last.sample;
        let tape_width: u64 = (x_sample * last.len()) as u64;
        let tape_min_index = last.tape_start();
        let x_actual: u32 = x_sample.divide_into(tape_width) as u32;
        let y_actual: u32 = self.spacelines.len() as u32;

        let row_bytes = x_actual;
        let mut packed_data = vec![0u8; row_bytes as usize * y_actual as usize];

        for y in 0..y_actual {
            let spaceline = self.spacelines.get(y as usize, &last);
            let local_start = &spaceline.tape_start();
            let local_x_sample = spaceline.sample;
            let local_per_x_sample = x_sample / local_x_sample;
            let row_start_byte_index: u32 = y * row_bytes;
            let x_start = x_sample.div_ceil_into(local_start - tape_min_index);
            for x in x_start as u32..x_actual {
                let tape_index: i64 = (x_sample * x as usize) as i64 + tape_min_index;
                // cmk consider changing asserts to debug_asserts
                assert!(
                    tape_index >= *local_start,
                    "real assert if x_start is correct"
                );

                let local_spaceline_start: i64 =
                    local_x_sample.divide_into(tape_index - local_start);

                // this helps medium bb6 go from 5 seconds to 3.5
                if local_per_x_sample == PowerOfTwo::ONE || self.x_smoothness == PowerOfTwo::ONE {
                    {
                        if local_spaceline_start >= spaceline.len() as i64 {
                            break;
                        }
                    }
                    let pixel = spaceline
                        .pixel_index_unbounded(local_spaceline_start as usize)
                        .0;
                    if pixel != 0 {
                        let byte_index: u32 = x + row_start_byte_index;
                        packed_data[byte_index as usize] = pixel;
                    }
                    continue;
                }
                // cmk LATER can we make this after by precomputing the collect outside the loop?
                let slice = (local_spaceline_start
                    ..local_spaceline_start + local_per_x_sample.as_u64() as i64)
                    .map(|i| spaceline.pixel_index_unbounded(i as usize))
                    .collect::<Vec<_>>();
                // cmk LATER look at putting this back in
                // if local_spaceline_index >= spaceline.pixels.len() as i64 {
                //     break;
                // }

                // Sample & Averaging 5 --
                let pixel = Pixel::merge_slice_all(&slice, 0).0;
                if pixel != 0 {
                    let byte_index: u32 = x + row_start_byte_index;
                    packed_data[byte_index as usize] = pixel;
                }
            }
        }

        encode_png(x_actual, y_actual, &packed_data)
    }
}
