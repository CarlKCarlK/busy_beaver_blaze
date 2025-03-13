use itertools::Itertools;

use crate::{
    Error, Machine, Pixel, PixelPolicy, SpaceByTimeMachine, Tape, encode_png, find_stride, is_even,
    power_of_two::PowerOfTwo, sample_rate, spaceline::Spaceline, spacelines::Spacelines,
};

#[derive(Clone)]
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
        // cmk0 confusing to refer to both machine time and space time as "step_count"
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

        // cmk000 instead of sample_rate, use find_stride???
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

    // cmk0 understand the `compress_cmk*` functions
    // cmk000 early_stop is just for auditing. Move it to be the last argument and rename it to early_stop_audit
    pub(crate) fn compress_cmk3_y_if_needed(&mut self, early_stop: Option<u64>) {
        let goal_y = self.y_goal;
        // cmk remove this variable
        let space_by_time = self;
        let binning = match space_by_time.pixel_policy {
            PixelPolicy::Binning => true,
            PixelPolicy::Sampling => false,
        };
        // cmk move audit_one to SpaceByTime
        SpaceByTimeMachine::audit_one(space_by_time, None, None, early_stop, binning);

        loop {
            SpaceByTimeMachine::audit_one(space_by_time, None, None, early_stop, binning);
            let len = space_by_time.spacelines.len();
            assert!(len > 0, "real assert 5");
            if len < goal_y as usize * 2 {
                break;
            }
            if !is_even(space_by_time.spacelines.main.len()) {
                SpaceByTimeMachine::audit_one(space_by_time, None, None, early_stop, binning);
                // println!("is odd: Spacelines {:?}", space_by_time.spacelines);
                let last = space_by_time.spacelines.main.pop().unwrap();
                space_by_time
                    .spacelines
                    .buffer0
                    .insert(0, (last, space_by_time.y_stride));
                SpaceByTimeMachine::audit_one(space_by_time, None, None, early_stop, binning);
            }
            // println!("Spacelines: {:?}", space_by_time.spacelines);

            assert!(
                is_even(space_by_time.spacelines.main.len()),
                "real assert 6"
            );

            SpaceByTimeMachine::audit_one(space_by_time, None, None, early_stop, binning);
            space_by_time.spacelines.main = space_by_time
                .spacelines
                .main
                .drain(..)
                .tuples()
                .map(|(mut first, second)| {
                    // println!("tuple a: {:?} b: {:?}", first.time, second.time);
                    assert!(first.tape_start() >= second.tape_start(), "real assert 4a");

                    if binning {
                        // cmk00 remove from loop?
                        first.merge(&second);
                    } else {
                        /* do nothing */
                    }
                    first
                })
                .collect();
            space_by_time.y_stride = space_by_time.y_stride.double();
            // println!("After binning or sampling: {:?}", space_by_time.spacelines);
            SpaceByTimeMachine::audit_one(space_by_time, None, None, early_stop, binning);
        }
    }

    // cmk ideas
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

        // cmk0 move this into a function
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

        let last = self.spacelines.last(self.y_stride, self.pixel_policy);

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

    // cmk000 works on packed_data that is one too big, currently can't use SIMD because packed data is not aligned
    // cmk0000move to lib, use aligned vec and slice
    fn compress_cmk4_y_if_needed(
        &self,
        mut packed_data: Vec<u8>,
        y_goal: u32,
        x_actual: u32,
        y_actual: u32,
    ) -> (Vec<u8>, u32) {
        if y_actual == 2 * y_goal {
            // cmk remove the constant
            // reduce the # of rows in half my averaging
            let mut new_packed_data = vec![0u8; x_actual as usize * y_goal as usize];
            packed_data
                .chunks_exact_mut(x_actual as usize * 2)
                .zip(new_packed_data.chunks_exact_mut(x_actual as usize))
                .for_each(|(chunk, new_chunk)| {
                    let (left, right) = chunk.split_at_mut(x_actual as usize);
                    // cmk00 so many issues: why no_simd?
                    // cmk00 why binning in the inner loop?
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

    pub(crate) fn reduce_buffer0(&mut self) {
        // cmk remove this variable
        let space_by_time = self;

        // take the buffer0 leaving it empty
        let buffer_old = core::mem::take(&mut space_by_time.spacelines.buffer0);
        let buffer0 = &mut space_by_time.spacelines.buffer0;
        let mut old_weight = None;
        for (spaceline, weight) in buffer_old {
            assert!(
                old_weight.is_none() || old_weight.unwrap() >= weight,
                "should be monotonic"
            );
            assert!(weight <= space_by_time.y_stride, "should be <= y_stride");
            if weight == space_by_time.y_stride {
                // This is a special case where we have a spaceline that is exactly the y_stride, so we can just push it to the main buffer
                space_by_time.spacelines.main.push(spaceline);
                continue;
            }
            Spacelines::push_internal(buffer0, spaceline, weight, space_by_time.pixel_policy);
            old_weight = Some(weight);

            if buffer0.len() == 1 && buffer0.first().unwrap().1 == space_by_time.y_stride {
                // This is a special case where we have a spaceline that is exactly the y_stride, so we can just push it to the main buffer
                let (spaceline_z, _weight_z) = buffer0.pop().unwrap();
                space_by_time.spacelines.main.push(spaceline_z);
            }
        }
    }

    pub(crate) fn extend(&mut self, other: Self) {
        // cmk00 remove this variable
        let y_stride = self.y_stride;
        let spacelines_other = other.spacelines;
        let main_other = spacelines_other.main;
        let buffer0_other = spacelines_other.buffer0;

        // If y_strides match, add other's main spacelines to the main buffer else add to the buffer0
        let (mut previous_y_stride, mut previous_time) = if other.y_stride == y_stride {
            for spaceline in main_other {
                self.spacelines.main.push(spaceline);
            }
            (self.y_stride, self.spacelines.main.last().unwrap().time)
        } else {
            for spaceline in main_other {
                self.spacelines.buffer0.push((spaceline, other.y_stride));
            }
            (
                other.y_stride,
                self.spacelines.buffer0.last().unwrap().0.time,
            )
        };

        // Add other's buffer0 spacelines to buffer0
        for (spaceline, weight) in buffer0_other {
            let time = spaceline.time;
            assert!(
                time == previous_time + previous_y_stride.as_u64(),
                "mind the gap"
            );
            previous_y_stride = weight;
            previous_time = time;
            self.spacelines.buffer0.push((spaceline, weight));
        }

        loop {
            self.compress_cmk3_y_if_needed(None);
            self.reduce_buffer0();

            if self.spacelines.len() <= self.y_goal as usize * 2 {
                break;
            }
        }
    }
}
