use crate::{
    Error, Machine, Pixel, PixelPolicy, Tape, encode_png, power_of_two::PowerOfTwo, sample_rate,
    spaceline::Spaceline, spacelines::Spacelines,
};

pub struct SpaceByTime {
    skip: u64,
    pub(crate) step_index: u64, // cmk make private
    x_goal: u32,
    y_goal: u32,
    pub(crate) y_stride: PowerOfTwo,   // cmk make private
    pub(crate) spacelines: Spacelines, // cmk0 consider making this private
    pixel_policy: PixelPolicy,
    previous_space_line: Option<Spaceline>,
}

/// Create a new in which you give the `x_goal` (space)
/// and the `y_goal` (time). The sample starts at 1 and
/// inner is a vector of one spaceline
impl SpaceByTime {
    #[inline]
    #[must_use]
    pub fn new0(x_goal: u32, y_goal: u32, pixel_policy: PixelPolicy) -> Self {
        Self {
            skip: 0,
            step_index: 0,
            x_goal,
            y_goal,
            y_stride: PowerOfTwo::ONE,
            spacelines: Spacelines::new0(pixel_policy),
            pixel_policy,
            previous_space_line: None,
        }
    }

    #[inline]
    #[must_use]
    pub fn new_skipped(
        tape: &Tape,
        skip: u64,
        x_goal: u32,
        y_goal: u32,
        pixel_policy: PixelPolicy,
    ) -> Self {
        // cmk000 confusing to refer to both machine time and space time as "step_count"
        Self {
            skip,
            step_index: 0,
            x_goal,
            y_goal,
            y_stride: PowerOfTwo::ONE,
            spacelines: Spacelines::new_skipped(tape, x_goal, skip, pixel_policy),
            pixel_policy,
            previous_space_line: None,
        }
    }

    #[inline]
    #[must_use]
    pub const fn step_index(&self) -> u64 {
        self.step_index
    }

    pub(crate) fn compress_if_needed(&mut self) {
        // cmk make private
        // Sampling & Averaging 1--
        // We sometimes need to squeeze rows by averaging adjacent pairs of rows.
        // The alternative is just to keep the 1st row and discard the 2nd row.

        let new_sample = sample_rate(self.step_index, self.y_goal);
        if new_sample != self.y_stride {
            assert!(
                new_sample / self.y_stride == PowerOfTwo::TWO,
                "real assert 10"
            );
            self.y_stride = new_sample;
            match self.pixel_policy {
                PixelPolicy::Binning => self.spacelines.compress_average(),
                PixelPolicy::Sampling => self.spacelines.compress_take_first(new_sample),
            }
        }
    }

    // ideas
    // use
    //       assert!(self.stride.is_power_of_two(), "Sample must be a power of two");
    //       // Use bitwise AND for fast divisibility check
    //       if self.step_index & (self.stride - 1) != 0 {
    //  Also: Inline the top part of the function.
    //  Maybe pre-subtract 1 from sample

    #[inline]
    pub(crate) fn snapshot(&mut self, machine: &Machine, previous_tape_index: i64) {
        self.step_index += 1;
        let inside_index = self.y_stride.rem_into_u64(self.step_index);

        match self.pixel_policy {
            PixelPolicy::Binning => {
                if inside_index == 0 {
                    // We're starting a new set of spacelines, so flush the buffer and compress (if needed)
                    self.spacelines.flush_buffer0();
                    self.compress_if_needed();
                }
                let spaceline = if let Some(mut previous) = self.previous_space_line.take() {
                    // cmk messy code
                    if previous.redo_pixel(
                        previous_tape_index,
                        machine.tape(),
                        self.x_goal,
                        self.step_index + self.skip,
                        self.pixel_policy,
                    ) {
                        previous
                    } else {
                        Spaceline::new(
                            machine.tape(),
                            self.x_goal,
                            self.step_index + self.skip,
                            self.pixel_policy,
                        )
                    }
                } else {
                    Spaceline::new(
                        machine.tape(),
                        self.x_goal,
                        self.step_index + self.skip,
                        self.pixel_policy,
                    )
                };
                self.previous_space_line = Some(spaceline.clone());
                self.spacelines.push(spaceline);
            }
            PixelPolicy::Sampling => {
                if inside_index != 0 {
                    return;
                }
                // We're starting a new set of spacelines, so flush the buffer and compress (if needed)
                self.spacelines.flush_buffer0();
                self.compress_if_needed();
                self.spacelines.push(Spaceline::new(
                    machine.tape(),
                    self.x_goal,
                    self.step_index + self.skip,
                    self.pixel_policy,
                ));
            }
        }
    }

    #[inline]
    pub(crate) fn push_spaceline(&mut self, spaceline: Spaceline, weight: PowerOfTwo) {
        let inside_index = self.y_stride.rem_into_u64(self.step_index);

        if inside_index == 0 {
            // We're starting a new set of spacelines, so flush the buffer and compress (if needed)
            self.spacelines.flush_buffer0();
            self.compress_if_needed();
        }

        if self.pixel_policy == PixelPolicy::Sampling && inside_index != 0 {
            return;
        }

        let buffer0 = &mut self.spacelines.buffer0;

        if buffer0.is_empty() {
            Spacelines::push_internal(buffer0, spaceline, weight);
            self.step_index += weight.as_u64();
            return;
        }
        let (last_spaceline, last_weight) = buffer0.last().unwrap();
        assert!(last_spaceline.time < spaceline.time, "real assert 11");
        if weight <= *last_weight {
            Spacelines::push_internal(buffer0, spaceline, weight);
            self.step_index += weight.as_u64();
            return;
        }

        println!(
            "last_spaceline's x stride {}, -len {}, +len {}",
            last_spaceline.x_stride.as_usize(),
            last_spaceline.negative.len(),
            last_spaceline.nonnegative.len()
        );
        println!(
            "spaceline's x stride {} -len {}, +len {}",
            spaceline.x_stride.as_usize(),
            spaceline.negative.len(),
            spaceline.nonnegative.len()
        );

        // cmk0000 should divide to bigger pieces
        println!(
            "last_weight {}, adding weight {} one by one",
            last_weight.as_u64(),
            weight.as_u64()
        );
        for i in 0..weight.as_u64() {
            let mut clone = spaceline.clone(); // cmk don't need to clone the last time
            clone.time = spaceline.time + i;
            if i % 100 == 0 {
                println!(
                    "clone's x stride {} -len {}, +len {}",
                    clone.x_stride.as_usize(),
                    clone.negative.len(),
                    clone.nonnegative.len()
                );
            }
            Spacelines::push_internal(buffer0, clone, PowerOfTwo::ONE);
            self.step_index += 1;
        }
    }

    #[allow(clippy::wrong_self_convention)] // cmk00 consider better name to this function
    pub fn to_png(&mut self) -> Result<Vec<u8>, Error> {
        let last = self
            .spacelines
            .last(self.step_index, self.y_stride, self.pixel_policy);
        let x_stride: PowerOfTwo = last.x_stride;
        let tape_width: u64 = (x_stride * last.len()) as u64;
        let tape_min_index = last.tape_start();
        let x_actual: u32 = x_stride.divide_into(tape_width) as u32;
        let y_actual: u32 = self.spacelines.len_with_01_buffer() as u32;

        let row_bytes = x_actual;
        let mut packed_data = vec![0u8; row_bytes as usize * y_actual as usize];

        for y in 0..y_actual {
            let spaceline = self.spacelines.get(y as usize, &last);
            let local_start = &spaceline.tape_start();
            let local_x_sample = spaceline.x_stride;
            let local_per_x_sample = x_stride / local_x_sample;
            let row_start_byte_index: u32 = y * row_bytes;
            let x_start = x_stride.div_ceil_into(local_start - tape_min_index);
            // cmk does the wrong thing with 1 row
            for x in x_start as u32..x_actual {
                let tape_index: i64 = (x_stride * x as usize) as i64 + tape_min_index;
                // cmk consider changing asserts to debug_asserts
                assert!(
                    tape_index >= *local_start,
                    "real assert if x_start is correct"
                );

                let local_spaceline_start: i64 =
                    local_x_sample.divide_into(tape_index - local_start);

                // this helps medium bb6 go from 5 seconds to 3.5
                if local_per_x_sample == PowerOfTwo::ONE
                    || self.pixel_policy == PixelPolicy::Sampling
                {
                    {
                        if local_spaceline_start >= spaceline.len() as i64 {
                            break;
                        }
                    }
                    let pixel_u8 =
                        u8::from(spaceline.pixel_index_unbounded(local_spaceline_start as usize));
                    if pixel_u8 != 0 {
                        let byte_index: u32 = x + row_start_byte_index;
                        packed_data[byte_index as usize] = pixel_u8;
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
                let pixel_u8 = u8::from(Pixel::merge_slice_all(&slice, 0));
                if pixel_u8 != 0 {
                    let byte_index: u32 = x + row_start_byte_index;
                    packed_data[byte_index as usize] = pixel_u8;
                }
            }
        }

        encode_png(x_actual, y_actual, &packed_data)
    }
}
