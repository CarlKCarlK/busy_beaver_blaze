use crate::{
    Error, Machine, Pixel, PixelPolicy, Tape, encode_png, find_stride,
    power_of_two::PowerOfTwo,
    sample_rate,
    spaceline::{self, Spaceline},
    spacelines::Spacelines,
};

pub struct SpaceByTime {
    skip: u64,
    pub(crate) step_index: u64,           // cmk make private
    pub(crate) x_goal: u32,               // cmk make private
    pub(crate) y_goal: u32,               // cmk make private
    pub(crate) y_stride: PowerOfTwo,      // cmk make private
    pub(crate) spacelines: Spacelines,    // cmk0 consider making this private
    pub(crate) pixel_policy: PixelPolicy, // cmk0 consider making this private
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

    pub(crate) fn compress_cmk1_y_if_needed(&mut self) {
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
                PixelPolicy::Binning => self.spacelines.compress_y_average(),
                PixelPolicy::Sampling => self.spacelines.compress_y_take_first(new_sample),
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
                    self.compress_cmk1_y_if_needed();
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
                self.spacelines
                    .push(spaceline, self.pixel_policy, PowerOfTwo::ONE);
            }
            PixelPolicy::Sampling => {
                if inside_index != 0 {
                    return;
                }
                // We're starting a new set of spacelines, so flush the buffer and compress (if needed)
                self.spacelines.flush_buffer0();
                self.compress_cmk1_y_if_needed();
                self.spacelines.push(
                    Spaceline::new(
                        machine.tape(),
                        self.x_goal,
                        self.step_index + self.skip,
                        self.pixel_policy,
                    ),
                    self.pixel_policy,
                    self.y_stride,
                );
            }
        }
    }

    #[inline]
    pub(crate) fn push_spaceline(&mut self, spaceline: Spaceline, weight: PowerOfTwo) {
        let inside_index = self.y_stride.rem_into_u64(self.step_index);

        if inside_index == 0 {
            // We're starting a new set of spacelines, so flush the buffer and compress (if needed)
            self.spacelines.flush_buffer0();
            self.compress_cmk1_y_if_needed();
        }

        if self.pixel_policy == PixelPolicy::Sampling && inside_index != 0 {
            return;
        }

        let buffer0 = &mut self.spacelines.buffer0;

        if buffer0.is_empty() {
            Spacelines::push_internal(buffer0, spaceline, weight, self.pixel_policy);
            self.step_index += weight.as_u64();
            return;
        }
        let (last_spaceline, last_weight) = buffer0.last().unwrap();
        assert!(last_spaceline.time < spaceline.time, "real assert 11");
        if weight <= *last_weight {
            Spacelines::push_internal(buffer0, spaceline, weight, self.pixel_policy);
            self.step_index += weight.as_u64();
            return;
        }

        // println!(
        //     "last_spaceline's x stride {}, -len {}, +len {}",
        //     last_spaceline.x_stride.as_usize(),
        //     last_spaceline.negative.len(),
        //     last_spaceline.nonnegative.len()
        // );
        // println!(
        //     "spaceline's x stride {} -len {}, +len {}",
        //     spaceline.x_stride.as_usize(),
        //     spaceline.negative.len(),
        //     spaceline.nonnegative.len()
        // );

        // cmk0000 should divide to bigger pieces
        // println!(
        //     "last_weight {}, adding weight {} one by one",
        //     last_weight.as_u64(),
        //     weight.as_u64()
        // );
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
            Spacelines::push_internal(buffer0, clone, PowerOfTwo::ONE, self.pixel_policy);
            self.step_index += 1;
        }
    }

    pub fn to_png(
        &mut self,
        tape_negative_len: usize,
        tape_nonnegative_len: usize,
        x_goal: usize,
        y_goal: usize,
    ) -> Result<Vec<u8>, Error> {
        let (png, _x, _y, _packed_data) =
            self.to_png_and_packed_data(tape_negative_len, tape_nonnegative_len, x_goal, y_goal)?;
        Ok(png)
    }

    #[allow(
        clippy::wrong_self_convention,
        clippy::missing_panics_doc,
        clippy::too_many_lines,
        clippy::shadow_reuse // cmk turn this off globally
    )]
    pub fn to_png_and_packed_data(
        &mut self,
        tape_negative_len: usize,
        tape_nonnegative_len: usize,
        x_goal: usize,
        y_goal: usize,
    ) -> Result<(Vec<u8>, u32, u32, Vec<u8>), Error> {
        assert!(tape_nonnegative_len > 0);
        assert!(x_goal >= 2);
        // println!("to_png y_stride {:?}--{:?}", self.y_stride, self.spacelines);

        let x_stride = find_stride(tape_negative_len, tape_nonnegative_len, x_goal);
        let x_zero = x_stride.div_ceil_into(tape_negative_len);
        let x_actual = x_zero + x_stride.div_ceil_into(tape_nonnegative_len);

        let y_actual = self.spacelines.len();

        let mut packed_data = vec![0u8; x_actual * y_actual];
        // println!(
        //     "tape_nonnegative_len {tape_nonnegative_len}, packed_data ({x_actual},{y_actual}) {packed_data:?} x_zero {x_zero} x_stride {x_stride:?}"
        // );

        // cmk000 move this into a function
        for (spaceline, _weight) in &mut self.spacelines.buffer0 {
            if spaceline.x_stride == x_stride {
                break;
            }
            spaceline.compress_x_if_needed(x_stride);
        }
        for spaceline in &mut self.spacelines.main {
            if spaceline.x_stride == x_stride {
                break;
            }
            spaceline.compress_x_if_needed(x_stride);
        }

        let last = self
            .spacelines
            .last(self.step_index, self.y_stride, self.pixel_policy);

        // println!("last spaceline {last:?}");

        for y in 0..y_actual {
            let spaceline = self.spacelines.get(y, &last);
            assert!(x_stride == spaceline.x_stride);

            let row_start_byte_index_plus_zero = y * x_actual + x_zero;
            for (index, pixel) in spaceline.negative.iter().enumerate() {
                let pixel_u8 = u8::from(pixel);
                if pixel_u8 != 0 {
                    let byte_index = row_start_byte_index_plus_zero - index - 1;
                    packed_data[byte_index] = pixel_u8;
                }
            }
            for (index, pixel) in spaceline.nonnegative.iter().enumerate() {
                let pixel_u8 = u8::from(pixel);
                if pixel_u8 != 0 {
                    let byte_index = row_start_byte_index_plus_zero + index;
                    packed_data[byte_index] = pixel_u8;
                }
            }
        }

        // println!("packed_data ({x_actual},{y_actual}) {packed_data:?}");
        assert!(y_actual <= 2 * y_goal);
        let (packed_data, y_actual) = self.compress_cmk4_y_if_needed(
            packed_data,
            y_goal as u32,
            x_actual as u32,
            y_actual as u32,
        );
        // let (x_actual, y_actual) =
        //     Self::trim_columns(&mut packed_data, x_actual as usize, y_actual as usize);

        let png = encode_png(x_actual as u32, y_actual, &packed_data)?;

        Ok((png, x_actual as u32, y_actual, packed_data))
    }

    // fn trim_columns(matrix: &mut Vec<u8>, xs: usize, ys: usize) -> (usize, usize) {
    //     assert!(!matrix.is_empty());
    //     assert_eq!(xs * ys, matrix.len());

    //     let mut first_nonzero_col = None;
    //     let mut last_nonzero_col = None;

    //     // Find the first non-zero column.
    //     'outer_first: for x in 0..xs {
    //         for y in 0..ys {
    //             if matrix[y * xs + x] != 0 {
    //                 first_nonzero_col = Some(x);
    //                 break 'outer_first;
    //             }
    //         }
    //     }

    //     // Find the last non-zero column.
    //     'outer_last: for x in (0..xs).rev() {
    //         for y in 0..ys {
    //             if matrix[y * xs + x] != 0 {
    //                 last_nonzero_col = Some(x);
    //                 break 'outer_last;
    //             }
    //         }
    //     }

    //     // Use `let else` to handle the case where no nonzero column was found (all zeros).
    //     let (Some(first_col), Some(last_col)) = (first_nonzero_col, last_nonzero_col) else {
    //         // Entire matrix is zeros: return a single-column matrix of zeros.
    //         matrix.truncate(ys); // Keep the first element of each row.
    //         return (1, ys);
    //     };

    //     // If no trimming is needed (i.e. every column has a nonzero), return unchanged.
    //     if first_col == 0 && last_col == xs - 1 {
    //         return (xs, ys);
    //     }

    //     let new_cols = last_col - first_col + 1;

    //     // Move data in-place row by row.
    //     for y in 0..ys {
    //         let src_start = y * xs + first_col;
    //         let dst_start = y * new_cols;
    //         for i in 0..new_cols {
    //             matrix[dst_start + i] = matrix[src_start + i];
    //         }
    //     }

    //     // Truncate any excess elements.
    //     matrix.truncate(ys * new_cols);
    //     (new_cols, ys)
    // }

    fn compress_cmk4_y_if_needed(
        &mut self,
        mut packed_data: Vec<u8>,
        y_goal: u32,
        x_actual: u32,
        y_actual: u32,
    ) -> (Vec<u8>, u32) {
        if y_actual == 2 * y_goal {
            // cmk0000000 remove the constant
            // reduce the # of rows in half my averaging
            let mut new_packed_data = vec![0u8; x_actual as usize * y_goal as usize];
            packed_data
                .chunks_exact_mut(x_actual as usize * 2)
                .zip(new_packed_data.chunks_exact_mut(x_actual as usize))
                .for_each(|(chunk, new_chunk)| {
                    let (left, right) = chunk.split_at_mut(x_actual as usize);
                    // cmk0000 so many issues: why no_simd? why binning in the inner loop?
                    match self.pixel_policy {
                        PixelPolicy::Binning => Pixel::slice_merge_bytes_no_simd(left, right),
                        PixelPolicy::Sampling => { /* do nothing */ }
                    }

                    // by design new_chunk is the same size as left, so copy the bytes from left to new_chunk
                    new_chunk.copy_from_slice(left);
                });
            // println!(
            //     "new_packed_data ({x_actual},{}) {new_packed_data:?}",
            //     y_actual >> 1
            // );
            (new_packed_data, y_goal)
        } else {
            (packed_data, y_actual)
        }
    }
}
