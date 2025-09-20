use core::num::NonZeroU8;
#[cfg(feature = "simd")]
use core::simd::{LaneCount, Simd, SupportedLaneCount, prelude::*};

use crate::pixel::Pixel;
use crate::symbol::Symbol;
use crate::tape::Tape;
use crate::test_utils::{compress_x_no_simd_binning, compress_x_no_simd_sampling};
use crate::{ALIGN, PixelPolicy, PowerOfTwo};
use aligned_vec::AVec;
#[cfg(feature = "simd")]
use zerocopy::IntoBytes;

#[derive(Clone)]
pub struct Spaceline {
    pub select: NonZeroU8,
    pub x_stride: PowerOfTwo,
    pub negative: AVec<Pixel>,
    pub nonnegative: AVec<Pixel>,
    pub time: u64,
    pub pixel_policy: PixelPolicy,
}

// define a Debug trait for Spaceline that lists x_stride, time, pixel_policy
// and for shows the negatives in reverse order with just the u8 values then "|" and the nonnegatives
// in order with just the u8 values
impl core::fmt::Debug for Spaceline {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let negative_str = self
            .negative
            .iter()
            .rev()
            .map(|pixel: &Pixel| pixel.as_u8().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let nonnegative_str = self
            .nonnegative
            .iter()
            .map(|pixel: &Pixel| pixel.as_u8().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            formatter,
            "Spaceline {{ Pixels: {} | {}, x_stride: {:?}, time: {}, pixel_policy: {:?}}}",
            negative_str, nonnegative_str, self.x_stride, self.time, self.pixel_policy
        )
    }
}

impl Spaceline {
    // This is complicated because the tape is in two parts and we always start at 0.
    #[allow(clippy::missing_panics_doc)]
    #[inline]
    #[must_use]
    pub fn find_x_stride(
        tape_neg_len: usize,
        tape_non_neg_len: usize,
        x_goal: usize,
    ) -> PowerOfTwo {
        assert!(x_goal >= 2, "Goal must be at least 2");
        let tape_len = tape_neg_len + tape_non_neg_len;
        // If the total length is less than the goal, use no downsampling.
        if tape_len < x_goal {
            return PowerOfTwo::ONE;
        }
        for exp in 0u8..63 {
            let stride = PowerOfTwo::from_exp(exp);
            // Compute ceiling division for each tape.
            let neg_len = stride.div_ceil_into(tape_neg_len);
            let non_neg_len = stride.div_ceil_into(tape_non_neg_len);
            let len = neg_len + non_neg_len;
            // We want combined to be in [goal_x, 2*goal_x).
            if x_goal <= len && len < 2 * x_goal {
                return stride;
            }
        }
        panic!("x_stride not found. This should never happen. Please report this as a bug.",)
    }

    #[must_use]
    pub fn average_with_iterators(
        select: NonZeroU8,
        values: &AVec<Symbol>,
        step: PowerOfTwo,
    ) -> AVec<Pixel> {
        let mut result: AVec<Pixel, _> =
            AVec::with_capacity(ALIGN, step.div_ceil_into(values.len()));

        // Process complete chunks
        let chunk_iter = values.chunks_exact(step.as_usize());
        let remainder = chunk_iter.remainder();

        for chunk in chunk_iter {
            let sum: u32 = chunk
                .iter()
                .map(|symbol| symbol.select_to_u32(select))
                .sum();
            let average = step.divide_into(sum * 255).into();
            result.push(average);
        }

        // Handle the remainder - pad with zeros
        if !remainder.is_empty() {
            let sum: u32 = remainder
                .iter()
                .map(|symbol| symbol.select_to_u32(select))
                .sum();
            // We need to divide by step size, not remainder.len()
            let average = step.divide_into(sum * 255).into();
            result.push(average);
        }

        result
    }

    #[must_use]
    pub fn sample_with_iterators(
        select: NonZeroU8,
        values: &AVec<Symbol>,
        step: PowerOfTwo,
    ) -> AVec<Pixel> {
        let mut result: AVec<Pixel, _> =
            AVec::with_capacity(ALIGN, step.div_ceil_into(values.len()));

        // Process complete chunks
        let chunk_iter = values.chunks_exact(step.as_usize());
        let remainder = chunk_iter.remainder();

        for chunk in chunk_iter {
            result.push(Pixel::from_symbol(chunk[0], select));
        }

        if !remainder.is_empty() {
            result.push(Pixel::from_symbol(remainder[0], select));
        }

        result
    }

    // TODO move this to tape and give a better name (may no longer apply)
    #[cfg(feature = "simd")]
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn average_with_simd<const LANES: usize>(
        select: NonZeroU8,
        values: &AVec<Symbol>,
        step: PowerOfTwo,
    ) -> AVec<Pixel>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        debug_assert!(
            { LANES } <= step.as_usize() && { LANES } <= { ALIGN },
            "LANES must be less than or equal to step and alignment"
        );

        let values_len = values.len();
        let capacity = step.div_ceil_into(values_len);

        // Pre-fill `result` with white to avoid `push()`
        let mut result: AVec<Pixel, _> = AVec::with_capacity(ALIGN, capacity);
        result.resize(capacity, Pixel::WHITE);

        let exp = LANES.trailing_zeros();
        let lanes_exp: u8 = exp.try_into().unwrap();
        let lanes = PowerOfTwo::from_exp(lanes_exp);
        let slice_u8 = values.as_slice().as_bytes();
        let (prefix, chunks, _suffix) = slice_u8.as_simd::<LANES>();

        debug_assert!(prefix.is_empty(), "Expected empty prefix due to alignment");
        let lanes_per_chunk = step.saturating_div(lanes);

        let select_vec = Simd::splat(select.get());

        if lanes_per_chunk == PowerOfTwo::ONE {
            for (average, chunk) in result.iter_mut().zip(chunks.iter()) {
                let sum = chunk.simd_eq(select_vec).to_bitmask().count_ones();
                *average = step.divide_into(sum * 255).into();
            }
        } else {
            let mut chunk_iter = chunks.chunks_exact(lanes_per_chunk.as_usize());
            for (average, sub_chunk) in result.iter_mut().zip(&mut chunk_iter) {
                let sum: u32 = sub_chunk
                    .iter()
                    .map(|chunk| chunk.simd_eq(select_vec).to_bitmask().count_ones())
                    .sum();
                *average = step.divide_into(sum * 255).into();
            }
        }

        // Efficiently handle remaining elements without `push()`
        let unused_items = step.rem_into_usize(values_len);
        if unused_items > 0 {
            let sum: u32 = values[values_len - unused_items..]
                .iter()
                .map(|symbol| symbol.select_to_u32(select))
                .sum();
            if let Some(last) = result.last_mut() {
                *last = step.divide_into(sum * 255).into();
            }
        }

        result
    }

    #[cfg(feature = "simd")]
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn average_chunk_with_simd<const LANES: usize>(
        select: NonZeroU8,
        chunk: &[Symbol],
        step: PowerOfTwo,
    ) -> Pixel
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        debug_assert!(
            { LANES } <= step.as_usize() && { LANES } <= { ALIGN },
            "LANES must be less than or equal to step and alignment"
        );
        debug_assert!(
            chunk.len() == step.as_usize(),
            "Chunk must be {} bytes",
            step.as_usize()
        );
        let (prefix, sub_chunks, suffix) = chunk.as_bytes().as_simd::<LANES>();

        debug_assert!(
            prefix.is_empty() && suffix.is_empty(),
            "Expected empty prefix due to alignment"
        );
        let exp = LANES.trailing_zeros();
        let lanes_exp: u8 = exp.try_into().unwrap();
        let lanes = PowerOfTwo::from_exp(lanes_exp);
        let lanes_per_chunk = step.saturating_div(lanes);
        debug_assert!(step.divides_usize(chunk.len()));

        let select_vec = Simd::splat(select.get());

        if lanes_per_chunk == PowerOfTwo::ONE {
            debug_assert!(sub_chunks.len() == 1, "Expected one chunk");
            let sum = sub_chunks[0].simd_eq(select_vec).to_bitmask().count_ones();
            step.divide_into(sum * 255).into()
        } else {
            let sum: u32 = sub_chunks
                .iter()
                .map(|sub_chunk| sub_chunk.simd_eq(select_vec).to_bitmask().count_ones())
                .sum();
            step.divide_into(sum * 255).into()
        }
    }

    #[inline]
    #[must_use]
    pub fn average_chunk_with_iterator(
        select: core::num::NonZeroU8,
        slice: &[Symbol],
        x_stride: PowerOfTwo,
    ) -> Pixel {
        debug_assert!(
            slice.len() == x_stride.as_usize(),
            "Chunk must be {} bytes",
            x_stride.as_usize()
        );

        let sum: u32 = slice
            .iter()
            .map(|symbol| symbol.select_to_u32(select))
            .sum();

        x_stride.divide_into(sum * 255).into()
    }

    #[must_use]
    #[inline]
    pub fn new0(select: NonZeroU8, pixel_policy: PixelPolicy) -> Self {
        let mut nonnegative = AVec::new(ALIGN);
        nonnegative.push(Pixel::WHITE);
        Self {
            select,
            x_stride: PowerOfTwo::ONE,
            negative: AVec::new(ALIGN),
            nonnegative,
            time: 0,
            pixel_policy,
        }
    }

    #[inline]
    #[must_use]
    pub fn pixel_index(&self, index: usize) -> Pixel {
        let negative_len = self.negative.len();
        if index < negative_len {
            self.negative[negative_len - 1 - index]
        } else {
            self.nonnegative[index - negative_len]
        }
    }

    #[inline]
    #[must_use]
    pub fn pixel_index_unbounded(&self, index: usize) -> Pixel {
        let negative_len = self.negative.len();
        if index < negative_len {
            self.negative
                .get(negative_len - 1 - index)
                .copied()
                .unwrap_or_default()
        } else {
            self.nonnegative
                .get(index - negative_len)
                .copied()
                .unwrap_or_default()
        }
    }

    #[inline]
    pub fn pixel_index_mut(&mut self, index: usize) -> &mut Pixel {
        let negative_len = self.negative.len();
        if index < negative_len {
            &mut self.negative[negative_len - 1 - index]
        } else {
            &mut self.nonnegative[index - negative_len]
        }
    }

    // This lets us compare the start of spaceline's with different x_strides.
    #[inline]
    #[must_use]
    /// Returns the starting tape index (in symbols) represented by the first pixel of this spaceline.
    ///
    /// # Panics
    /// Panics if the product `self.x_stride * self.negative.len()` does not fit in an `i64`.
    pub fn tape_start(&self) -> i64 {
        let width = self.x_stride * self.negative.len();
        let width = i64::try_from(width).expect("tape_start width must fit in i64");
        -width
    }

    #[cfg(feature = "simd")]
    #[allow(clippy::missing_panics_doc, clippy::shadow_reuse)]
    pub fn compress_x_simd_binning(pixels: &mut AVec<Pixel>) {
        // Constant used for the right shift in our average formula.
        const SPLAT_1: Simd<u8, ALIGN> = Simd::splat(1);
        let pixels_len = pixels.len();

        // Loop through an even number of 32-bit chunks
        let pixels_bytes = pixels.as_mut_slice().as_mut_bytes();
        let (prefix, chunks, _) = pixels_bytes.as_simd_mut::<ALIGN>();
        assert!(prefix.is_empty(), "assert aligned");
        let mut len_half = chunks.len() >> 1;
        let mut read_index = 0;
        let mut write_index = 0;
        while write_index < len_half {
            chunks[write_index] = {
                let (mut left, right) = chunks[read_index].deinterleave(chunks[read_index + 1]);
                read_index += 2;

                // Compute the average using the formula: (a & b) + ((a ^ b) >> 1)
                let mut result = left ^ right;
                result >>= SPLAT_1;
                left &= right;
                result += left;
                result
            };
            write_index += 1;
        }

        // Switch from chunks to bytes and finish any remaining chunks and the suffix.
        write_index *= ALIGN;
        read_index *= ALIGN;
        len_half = pixels_len >> 1;
        while write_index < len_half {
            pixels_bytes[write_index] = {
                let mut left = pixels_bytes[read_index];
                read_index += 1;
                let right = pixels_bytes[read_index];
                read_index += 1;

                // Compute the average using the formula: (a & b) + ((a ^ b) >> 1)
                let mut result = left ^ right;
                result >>= 1;
                left &= right;
                result += left;
                result
            };
            write_index += 1;
        }

        if read_index < pixels_len {
            // Handle the last pixel if the length is odd by reducing it by half
            pixels_bytes[write_index] = pixels_bytes[read_index] >> 1;
            // read_index += 1;
            write_index += 1;
        }
        pixels.truncate(write_index);
    }

    #[cfg(feature = "simd")]
    #[allow(clippy::shadow_reuse, clippy::missing_panics_doc)]
    pub fn compress_x_simd_sampling(pixels: &mut AVec<Pixel>) {
        let pixels_len = pixels.len();

        // Loop through an even number of 32-bit chunks
        let pixels_bytes = pixels.as_mut_slice().as_mut_bytes();
        let (prefix, chunks, _) = pixels_bytes.as_simd_mut::<ALIGN>();
        assert!(prefix.is_empty(), "assert aligned");
        let mut len_half = chunks.len() >> 1;
        let mut read_index = 0;
        let mut write_index = 0;
        while write_index < len_half {
            chunks[write_index] = {
                let (left, _) = chunks[read_index].deinterleave(chunks[read_index + 1]);
                read_index += 2;
                left
            };
            write_index += 1;
        }

        // Switch from chunks to bytes and finish any remaining chunks and the suffix.
        write_index *= ALIGN;
        read_index *= ALIGN;
        len_half = pixels_len >> 1;
        while write_index < len_half {
            pixels_bytes[write_index] = pixels_bytes[read_index];
            read_index += 2;
            write_index += 1;
        }

        if read_index < pixels_len {
            pixels_bytes[write_index] = pixels_bytes[read_index];
            // read_index += 1;
            write_index += 1;
        }
        pixels.truncate(write_index);
    }

    #[cfg(feature = "simd")]
    #[allow(clippy::missing_panics_doc)]
    #[inline]
    pub fn compress_x_if_needed_simd(&mut self, new_x_stride: PowerOfTwo) {
        // Sampling & Averaging 2 --
        assert!(self.x_stride <= new_x_stride);
        while self.x_stride < new_x_stride {
            for pixels in [&mut self.nonnegative, &mut self.negative] {
                // TODO pull this out of the inner loop
                match self.pixel_policy {
                    PixelPolicy::Binning => Self::compress_x_simd_binning(pixels),
                    PixelPolicy::Sampling => Self::compress_x_simd_sampling(pixels),
                }
            }
            self.x_stride = self.x_stride.double();
        }
    }

    #[allow(clippy::missing_panics_doc)]
    #[inline]
    pub fn compress_x_if_needed_no_simd(&mut self, new_x_stride: PowerOfTwo) {
        // Sampling & Averaging 2 --
        assert!(self.x_stride <= new_x_stride);
        while self.x_stride < new_x_stride {
            for pixels in [&mut self.nonnegative, &mut self.negative] {
                // TODO pull this out of the inner loop
                match self.pixel_policy {
                    PixelPolicy::Binning => compress_x_no_simd_binning(pixels),
                    PixelPolicy::Sampling => compress_x_no_simd_sampling(pixels),
                }
            }
            self.x_stride = self.x_stride.double();
        }
    }

    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn pixel_range(&self, start: usize, end: usize) -> Vec<Pixel> {
        assert!(
            start <= end,
            "start index {start} must be <= end index {end}"
        );
        let mut result = Vec::with_capacity(end - start);
        for i in start..end {
            result.push(self.pixel_index(i));
        }
        result
    }

    #[cfg(feature = "simd")]
    #[allow(clippy::missing_panics_doc)]
    #[inline]
    pub fn merge_simd(&mut self, other: &Self) {
        assert!(self.time < other.time,);
        assert!(self.x_stride <= other.x_stride);

        self.compress_x_if_needed_simd(other.x_stride);
        assert!(self.x_stride == other.x_stride);

        Pixel::avec_merge_simd(&mut self.negative, &other.negative);
        Pixel::avec_merge_simd(&mut self.nonnegative, &other.nonnegative);
    }

    #[allow(clippy::missing_panics_doc)]
    #[inline]
    pub fn merge_no_simd(&mut self, other: &Self) {
        assert!(self.time < other.time,);
        assert!(self.x_stride <= other.x_stride);

        self.compress_x_if_needed_no_simd(other.x_stride);
        assert!(self.x_stride == other.x_stride);

        Pixel::avec_merge_no_simd(&mut self.negative, &other.negative);
        Pixel::avec_merge_no_simd(&mut self.nonnegative, &other.nonnegative);
    }

    #[inline]
    pub fn merge_with_white_no_simd(&mut self) {
        Pixel::slice_merge_with_white_no_simd(&mut self.nonnegative);
        Pixel::slice_merge_with_white_no_simd(&mut self.negative);
    }

    #[cfg(feature = "simd")]
    #[inline]
    pub fn merge_with_white_simd(&mut self) {
        Pixel::slice_merge_with_white_simd(&mut self.nonnegative);
        Pixel::slice_merge_with_white_simd(&mut self.negative);
    }

    #[allow(clippy::missing_panics_doc)]
    #[inline]
    #[must_use]
    pub fn new(
        select: NonZeroU8,
        tape: &Tape,
        x_goal: u32,
        step_index: u64,
        pixel_policy: PixelPolicy,
    ) -> Self {
        // TODO move this to tape and give a better name
        let x_stride =
            Self::find_x_stride(tape.negative.len(), tape.nonnegative.len(), x_goal as usize);
        match pixel_policy {
            PixelPolicy::Binning => {
                #[cfg(not(feature = "simd"))] // cmk000 OK
                let (negative, nonnegative) =
                    // TODO move this to tape and give a better name
                    (
                        Self::average_with_iterators(select, &tape.negative, x_stride),
                        Self::average_with_iterators(select, &tape.nonnegative, x_stride),
                    );
                #[cfg(feature = "simd")]
                let (negative, nonnegative) = match x_stride {
                    PowerOfTwo::ONE | PowerOfTwo::TWO | PowerOfTwo::FOUR => (
                        // TODO move this to tape and give a better name
                        Self::average_with_iterators(select, &tape.negative, x_stride),
                        Self::average_with_iterators(select, &tape.nonnegative, x_stride),
                    ),
                    PowerOfTwo::EIGHT => (
                        // TODO move this to tape and give a better name
                        Self::average_with_simd::<8>(select, &tape.negative, x_stride),
                        Self::average_with_simd::<8>(select, &tape.nonnegative, x_stride),
                    ),
                    PowerOfTwo::SIXTEEN => (
                        Self::average_with_simd::<16>(select, &tape.negative, x_stride),
                        Self::average_with_simd::<16>(select, &tape.nonnegative, x_stride),
                    ),
                    PowerOfTwo::THIRTY_TWO => (
                        Self::average_with_simd::<32>(select, &tape.negative, x_stride),
                        Self::average_with_simd::<32>(select, &tape.nonnegative, x_stride),
                    ),
                    _ => (
                        Self::average_with_simd::<64>(select, &tape.negative, x_stride),
                        Self::average_with_simd::<64>(select, &tape.nonnegative, x_stride),
                    ),
                };
                Self {
                    select,
                    x_stride,
                    negative,
                    nonnegative,
                    time: step_index,
                    pixel_policy,
                }
            }
            PixelPolicy::Sampling => {
                let negative = Self::sample_with_iterators(select, &tape.negative, x_stride);
                let nonnegative = Self::sample_with_iterators(select, &tape.nonnegative, x_stride);
                Self {
                    select,
                    x_stride,
                    negative,
                    nonnegative,
                    time: step_index,
                    pixel_policy,
                }
            }
        }
    }

    #[inline]
    /// Recomputes one pixel corresponding to the last tape change, if possible.
    ///
    /// # Panics
    /// Panics if any computed index does not fit in the target integer type when converting:
    /// - Converting the previous negative tape index magnitude to `u64`.
    /// - Converting a non-negative `previous_tape_index` to `u64`.
    /// - Converting the computed `pixel_index` to `usize`.
    pub fn redo_pixel(
        &mut self,
        previous_tape_index: i64,
        tape: &Tape,
        x_goal: u32,
        step_index: u64,
        pixel_policy: PixelPolicy,
    ) -> bool {
        // let x_stride = sample_rate(tape.nonnegative.len() as u64, x_goal)
        //     .min(sample_rate(tape.negative.len() as u64, x_goal));
        let x_stride =
            Self::find_x_stride(tape.negative.len(), tape.nonnegative.len(), x_goal as usize);

        // When the sample
        if self.x_stride != x_stride {
            return false;
        }
        self.time = step_index;
        let (part, pixels, part_index) = if previous_tape_index < 0 {
            let idx = (-previous_tape_index) - 1;
            let part_index =
                u64::try_from(idx).expect("negative tape index magnitude must fit in u64");
            (&tape.negative, &mut self.negative, part_index)
        } else {
            let part_index =
                u64::try_from(previous_tape_index).expect("tape index must be non-negative u64");
            (&tape.nonnegative, &mut self.nonnegative, part_index)
        };
        let pixel_index = x_stride.divide_into(part_index);
        let pixel_index = usize::try_from(pixel_index).expect("pixel index must fit in usize");
        debug_assert!(pixel_index <= pixels.len());
        if pixel_index == pixels.len() {
            pixels.push(Pixel::WHITE);
        }
        let pixel = &mut pixels[pixel_index];
        match pixel_policy {
            PixelPolicy::Binning => {
                let tape_slice_start = x_stride * pixel_index;
                let tape_slice_end = tape_slice_start + x_stride.as_usize();
                // If tape is short, then sum one by one
                if part.len() < tape_slice_end {
                    let sum: u32 = (tape_slice_start..part.len())
                        .map(|i| part[i].select_to_u32(self.select))
                        .sum();
                    *pixel = Pixel::from(x_stride.divide_into(sum * 255));
                } else {
                    let slice = &part[tape_slice_start..tape_slice_end];

                    #[cfg(not(feature = "simd"))] // cmk000 OK
                    {
                        *pixel = Self::average_chunk_with_iterator(self.select, slice, x_stride);
                    }
                    #[cfg(feature = "simd")]
                    {
                        *pixel = match x_stride {
                            PowerOfTwo::ONE | PowerOfTwo::TWO | PowerOfTwo::FOUR => {
                                let sum: u32 = slice
                                    .iter()
                                    .map(|symbol| symbol.select_to_u32(self.select))
                                    .sum();
                                x_stride.divide_into(sum * 255).into()
                            }
                            PowerOfTwo::EIGHT => {
                                Self::average_chunk_with_simd::<8>(self.select, slice, x_stride)
                            }
                            PowerOfTwo::SIXTEEN => {
                                Self::average_chunk_with_simd::<16>(self.select, slice, x_stride)
                            }
                            PowerOfTwo::THIRTY_TWO => {
                                Self::average_chunk_with_simd::<32>(self.select, slice, x_stride)
                            }
                            _ => Self::average_chunk_with_simd::<64>(self.select, slice, x_stride),
                        };
                    }
                }
            }
            PixelPolicy::Sampling => {
                *pixel = Pixel::from_symbol(part[x_stride * pixel_index], self.select);
            }
        }
        true
    }
}
