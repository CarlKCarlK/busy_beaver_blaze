use aligned_vec::AVec;
use itertools::Itertools;

use crate::{
    ALIGN, Error, Machine, PixelPolicy, Tape, compress_packed_data_if_one_too_big, encode_png,
    find_x_stride, find_y_stride, is_even, power_of_two::PowerOfTwo, spaceline::Spaceline,
    spacelines::Spacelines,
};

#[derive(Clone)]
pub struct SpaceByTime {
    pub(crate) skip: u64,
    pub(crate) vis_step: u64,             // TODO make private
    pub(crate) x_goal: u32,               // TODO make private
    pub(crate) y_goal: u32,               // TODO make private
    pub(crate) y_stride: PowerOfTwo,      // TODO make private
    pub(crate) spacelines: Spacelines,    // TODO consider making this private
    pub(crate) pixel_policy: PixelPolicy, // TODO consider making this private
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
            vis_step: 0,
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
        Self {
            skip,
            vis_step: 0,
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
        self.vis_step + self.skip
    }

    #[inline]
    pub(crate) fn snapshot(&mut self, machine: &Machine, previous_tape_index: i64) {
        self.vis_step += 1;
        let inside_index = self.y_stride.rem_into_u64(self.vis_step);

        match self.pixel_policy {
            PixelPolicy::Binning => {
                if inside_index == 0 {
                    // We're starting a new set of spacelines, so flush the buffer and compress (if needed)
                    self.flush_buffer0_and_compress();
                }
                let spaceline = if let Some(mut previous) = self.previous_space_line.take() {
                    // TODO messy code
                    if previous.redo_pixel(
                        previous_tape_index,
                        machine.tape(),
                        self.x_goal,
                        self.step_index(),
                        self.pixel_policy,
                    ) {
                        previous
                    } else {
                        Spaceline::new(
                            machine.tape(),
                            self.x_goal,
                            self.step_index(),
                            self.pixel_policy,
                        )
                    }
                } else {
                    Spaceline::new(
                        machine.tape(),
                        self.x_goal,
                        self.step_index(),
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
                self.flush_buffer0_and_compress();
                self.spacelines.push(
                    Spaceline::new(
                        machine.tape(),
                        self.x_goal,
                        self.step_index(),
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
        zero_color: [u8; 3],
        one_color: [u8; 3],
    ) -> Result<Vec<u8>, Error> {
        let (png, _x, _y, _packed_data) = self.to_png_and_packed_data(
            tape_negative_len,
            tape_nonnegative_len,
            x_goal,
            y_goal,
            zero_color,
            one_color,
        )?;

        Ok(png)
    }

    #[allow(
        clippy::wrong_self_convention,
        clippy::missing_panics_doc,
        clippy::too_many_lines,
        clippy::shadow_reuse // TODO turn this off globally
    )]
    pub fn to_png_and_packed_data(
        &mut self,
        tape_negative_len: usize,
        tape_nonnegative_len: usize,
        x_goal: usize,
        y_goal: usize,
        zero_color: [u8; 3],
        one_color: [u8; 3],
    ) -> Result<(Vec<u8>, u32, u32, AVec<u8>), Error> {
        assert!(tape_nonnegative_len > 0);
        assert!(x_goal >= 2);
        // println!("to_png y_stride {:?}--{:?}", self.y_stride, self.spacelines);

        let x_stride = find_x_stride(tape_negative_len, tape_nonnegative_len, x_goal);
        let x_zero = x_stride.div_ceil_into(tape_negative_len);
        let x_actual = x_zero + x_stride.div_ceil_into(tape_nonnegative_len);

        let y_actual = self.spacelines.len();

        let mut packed_data = AVec::with_capacity(ALIGN, x_actual * y_actual);
        packed_data.resize(x_actual * y_actual, 0u8);

        // TODO too many lines, move this into a function
        for (spaceline, _weight) in &mut self.spacelines.buffer0 {
            if spaceline.x_stride == x_stride {
                break;
            }
            spaceline.compress_x_if_needed_simd(x_stride);
        }
        for spaceline in &mut self.spacelines.main {
            if spaceline.x_stride == x_stride {
                break;
            }
            spaceline.compress_x_if_needed_simd(x_stride);
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

        assert!(y_actual <= 2 * y_goal);
        let (packed_data, y_actual) = compress_packed_data_if_one_too_big(
            packed_data,
            self.pixel_policy,
            y_goal as u32,
            x_actual as u32,
            y_actual as u32,
        );

        let png = encode_png(
            x_actual as u32,
            y_actual,
            zero_color,
            one_color,
            &packed_data,
        )?;

        Ok((png, x_actual as u32, y_actual, packed_data))
    }

    pub(crate) fn append(mut self, other: Self) -> Self {
        let y_stride = self.y_stride;

        let main_other = other.spacelines.main;
        let buffer0_other = other.spacelines.buffer0;

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
            self.append_internal_compress();

            if self.spacelines.len() <= self.y_goal as usize * 2 {
                break;
            }
        }
        self.vis_step += other.vis_step + 1;
        self
    }

    pub(crate) fn append_internal_compress(&mut self) {
        let goal_y = self.y_goal;
        self.audit();

        loop {
            self.audit();
            let len = self.spacelines.len();
            assert!(len > 0, "real assert 5");
            if len < goal_y as usize * 2 {
                break;
            }
            if !is_even(self.spacelines.main.len()) {
                self.audit();
                // println!("is odd: Spacelines {:?}", self.spacelines);
                let last = self.spacelines.main.pop().unwrap();
                self.spacelines.buffer0.insert(0, (last, self.y_stride));
                self.audit();
            }

            assert!(is_even(self.spacelines.main.len()), "real assert 6");

            self.audit();
            self.spacelines.main = self
                .spacelines
                .main
                .drain(..)
                .tuples()
                .map(|(mut first, second)| {
                    assert!(first.tape_start() >= second.tape_start());

                    // TODO remove from loop?
                    match self.pixel_policy {
                        PixelPolicy::Binning => first.merge_simd(&second),
                        PixelPolicy::Sampling => (),
                    }
                    first
                })
                .collect();
            self.y_stride = self.y_stride.double();
            self.audit();
        }

        // take the buffer0 leaving it empty
        let buffer_old = core::mem::take(&mut self.spacelines.buffer0);
        let buffer0 = &mut self.spacelines.buffer0;
        let mut old_weight = None;
        for (spaceline, weight) in buffer_old {
            assert!(
                old_weight.is_none() || old_weight.unwrap() >= weight,
                "should be monotonic"
            );
            assert!(weight <= self.y_stride, "should be <= y_stride");
            if weight == self.y_stride {
                // This is a special case where we have a spaceline that is exactly the y_stride, so we can just push it to the main buffer
                self.spacelines.main.push(spaceline);
                continue;
            }
            Spacelines::push_internal(buffer0, spaceline, weight, self.pixel_policy);
            old_weight = Some(weight);

            if let [(_, weight_z)] = buffer0.as_slice()
                && *weight_z == self.y_stride
            {
                // This is a special case where we have a spaceline that is exactly the y_stride, so we can just push it to the main buffer
                let (spaceline_z, _weight_z) = buffer0.pop().unwrap();
                self.spacelines.main.push(spaceline_z);
            }
        }
    }

    #[inline]
    #[allow(unused_variables, clippy::shadow_reuse)]
    pub(crate) fn audit(&self) {
        // on debug compiles call audit_one_internal otherwise do nothing
        #[cfg(debug_assertions)]
        {
            let mut previous_y_stride: Option<PowerOfTwo> = None;
            let mut previous_time: Option<u64> = None;

            let y_stride = self.y_stride;
            if let Some(previous_y_stride) = previous_y_stride {
                assert_eq!(
                    y_stride, previous_y_stride,
                    "from part to part, the stride should be the same"
                );
            }
            let spacelines = &self.spacelines;
            let main = &spacelines.main;
            for spaceline in main {
                if let Some(previous_time) = previous_time {
                    assert!(
                        spaceline.time == previous_time + y_stride.as_u64(),
                        "mind the gap"
                    );
                } else {
                    assert_eq!(spaceline.time, 0, "first spaceline should be 0");
                }
                previous_time = Some(spaceline.time);
                previous_y_stride = Some(y_stride);
            }
            for (spaceline, weight) in &spacelines.buffer0 {
                assert!(
                    spaceline.time == previous_time.unwrap() + previous_y_stride.unwrap().as_u64(),
                    "mind the gap"
                );
                assert!(
                    *weight <= previous_y_stride.unwrap(),
                    "should be <= previous_y_stride"
                );
                previous_time = Some(spaceline.time);
                previous_y_stride = Some(*weight);
            }
        }
    }

    pub(crate) fn flush_buffer0_and_compress(&mut self) {
        debug_assert!(self.spacelines.buffer0.len() <= 1);
        let spacelines = &mut self.spacelines;
        if !spacelines.buffer0.is_empty() {
            assert!(spacelines.buffer0.len() == 1);
            let (spaceline, weight) = spacelines.buffer0.pop().unwrap();
            assert!(weight == self.y_stride);
            spacelines.main.push(spaceline);
        }
        let y_stride = find_y_stride(self.vis_step, self.y_goal);
        if y_stride != self.y_stride {
            assert_eq!(y_stride, self.y_stride.double());
            self.y_stride = y_stride;
            match self.pixel_policy {
                PixelPolicy::Binning => self.spacelines.compress_y_average(),
                PixelPolicy::Sampling => self.spacelines.compress_y_take_first(y_stride),
            }
        }
    }
}
