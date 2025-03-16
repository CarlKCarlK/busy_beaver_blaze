use core::simd::{Simd, simd_swizzle};

use crate::pixel::Pixel;
use crate::tape::Tape;
use crate::{
    ALIGN, PixelPolicy, PowerOfTwo, average_chunk_with_simd, average_with_iterators,
    average_with_simd, find_x_stride, sample_with_iterators,
};
use aligned_vec::AVec;
use zerocopy::IntoBytes;

#[derive(Clone)]
pub struct Spaceline {
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
    #[must_use]
    #[inline]
    pub fn new0(pixel_policy: PixelPolicy) -> Self {
        let mut nonnegative = AVec::new(ALIGN);
        nonnegative.push(Pixel::WHITE);
        Self {
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
    pub fn tape_start(&self) -> i64 {
        -((self.x_stride * self.negative.len()) as i64)
    }

    // cmk0000
    #[allow(clippy::cast_ptr_alignment)]
    pub fn compress_x_simd_binning(pixels: &mut AVec<Pixel>) {
        // Local constant for the static swizzle indices.
        const SWIZZLE_INDICES: [usize; 64] = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44,
            46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27,
            29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
        ];
        // Constant used for the right shift in our average formula.
        const SPLAT_1_32: Simd<u8, 32> = Simd::splat(1);

        // Access the pixel data as a mutable byte slice.
        let pixels_bytes = pixels.as_mut_slice().as_mut_bytes();
        let len = pixels_bytes.len();
        // We'll process the input in 64-byte blocks.
        let num_chunks = len >> 6; // Shift right by 6 is equivalent to dividing by 64

        // Destination pointer: we'll pack results to the start of this buffer.
        let dst_ptr = pixels_bytes.as_mut_ptr();
        let mut write_index = 0;

        // Process each 64-byte block.
        for i in 0..num_chunks {
            let chunk_start = i * 64;
            // Load the 64-byte chunk from the source.
            let src_chunk_ptr = unsafe {
                pixels_bytes
                    .as_ptr()
                    .add(chunk_start)
                    .cast::<Simd<u8, 64>>()
            };
            let chunk = unsafe { *src_chunk_ptr };

            // Static swizzle to rearrange:
            //  - first 32 lanes: even-indexed bytes,
            //  - last 32 lanes: odd-indexed bytes.
            let swizzled: Simd<u8, 64> = simd_swizzle!(chunk, SWIZZLE_INDICES);

            // Split the swizzled vector into two halves (each 32 lanes).
            let (left, right) = unsafe {
                // Cast the 64-lane vector to a pointer to two consecutive 32-lane vectors.
                let ptr = (&raw const swizzled).cast::<Simd<u8, 32>>();
                (*ptr, *ptr.add(1))
            };

            // Compute the average using the formula: (a & b) + ((a ^ b) >> 1)
            // cmk0000 could make this unbiased
            let result: &mut Simd<u8, 32> =
                unsafe { &mut *dst_ptr.add(write_index).cast::<Simd<u8, 32>>() };
            *result = left & right;
            let mut avg = left;
            avg ^= right;
            avg >>= SPLAT_1_32;
            *result += avg;

            write_index += 32;
        }

        // Process the remaining bytes (suffix) with scalar code.
        let mut i = num_chunks * 64;
        while i + 1 < len {
            // cmk0000
            // Compute the average for a pair of pixels.
            pixels_bytes[write_index] = Pixel::mean_bytes(pixels_bytes[i], pixels_bytes[i + 1]);
            write_index += 1;
            i += 2;
        }
        if i < len {
            // For a leftover pixel, average it with white (i.e. divide by 2).
            // cmk0000
            pixels_bytes[write_index] = pixels_bytes[i] >> 1;
            write_index += 1;
        }

        // Finally, truncate the vector to the new length (in bytes).
        pixels.truncate(write_index);
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn compress_x_simd_sampling(pixels: &mut AVec<Pixel>) {
        // Local constant for the static swizzle indices.
        const SWIZZLE_INDICES: [usize; 64] = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44,
            46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27,
            29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
        ];
        // Access the pixel data as a mutable byte slice.
        let pixels_bytes = pixels.as_mut_slice().as_mut_bytes();
        let len = pixels_bytes.len();
        // We'll process the input in 64-byte blocks.
        let num_chunks = len >> 6; // Shift right by 6 is equivalent to dividing by 64

        // Destination pointer: we'll pack results to the start of this buffer.
        let dst_ptr = pixels_bytes.as_mut_ptr();
        let mut write_index = 0;

        // Process each 64-byte block.
        for i in 0..num_chunks {
            let chunk_start = i * 64;
            // Load the 64-byte chunk from the source.
            let src_chunk_ptr = unsafe {
                pixels_bytes
                    .as_ptr()
                    .add(chunk_start)
                    .cast::<Simd<u8, 64>>()
            };
            let chunk = unsafe { *src_chunk_ptr };

            // Static swizzle to rearrange:
            //  - first 32 lanes: even-indexed bytes,
            //  - last 32 lanes: odd-indexed bytes.
            let swizzled: Simd<u8, 64> = simd_swizzle!(chunk, SWIZZLE_INDICES);

            // Split the swizzled vector into two halves (each 32 lanes).
            let left = unsafe { *(&raw const swizzled).cast::<Simd<u8, 32>>() };

            let result: &mut Simd<u8, 32> =
                unsafe { &mut *dst_ptr.add(write_index).cast::<Simd<u8, 32>>() };
            *result = left;
            write_index += 32;
        }

        // Process the remaining bytes (suffix) with scalar code.
        let mut i = num_chunks * 64;
        while i + 1 < len {
            pixels_bytes[write_index] = pixels_bytes[i];
            write_index += 1;
            i += 2;
        }
        if i < len {
            pixels_bytes[write_index] = pixels_bytes[i];
            write_index += 1;
        }

        // Finally, truncate the vector to the new length (in bytes).
        pixels.truncate(write_index);
    }

    #[allow(clippy::missing_panics_doc)]
    #[inline]
    pub fn compress_x_if_needed_simd(&mut self, new_x_stride: PowerOfTwo) {
        // Sampling & Averaging 2 --
        assert!(self.x_stride <= new_x_stride);
        while self.x_stride < new_x_stride {
            for pixels in [&mut self.nonnegative, &mut self.negative] {
                // cmk00 pull this out of the inner loop
                match self.pixel_policy {
                    PixelPolicy::Binning => Self::compress_x_simd_binning(pixels),
                    PixelPolicy::Sampling => Self::compress_x_simd_sampling(pixels),
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

    // cmk0000
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

    // cmk0000
    #[inline]
    pub fn merge_with_white_simd(&mut self) {
        Pixel::slice_merge_with_white_simd(&mut self.nonnegative);
        Pixel::slice_merge_with_white_simd(&mut self.negative);
    }

    #[allow(clippy::missing_panics_doc)]
    #[inline]
    #[must_use]
    pub fn new(tape: &Tape, x_goal: u32, step_index: u64, pixel_policy: PixelPolicy) -> Self {
        // cmk move this to tape and give a better name
        let x_stride = find_x_stride(tape.negative.len(), tape.nonnegative.len(), x_goal as usize);
        match pixel_policy {
            PixelPolicy::Binning => {
                let (negative, nonnegative) = match x_stride {
                    PowerOfTwo::ONE | PowerOfTwo::TWO | PowerOfTwo::FOUR => (
                        // cmk move this to tape and give a better name
                        average_with_iterators(&tape.negative, x_stride),
                        average_with_iterators(&tape.nonnegative, x_stride),
                    ),
                    PowerOfTwo::EIGHT => (
                        // cmk move this to tape and give a better name
                        average_with_simd::<8>(&tape.negative, x_stride),
                        average_with_simd::<8>(&tape.nonnegative, x_stride),
                    ),
                    PowerOfTwo::SIXTEEN => (
                        average_with_simd::<16>(&tape.negative, x_stride),
                        average_with_simd::<16>(&tape.nonnegative, x_stride),
                    ),
                    PowerOfTwo::THIRTY_TWO => (
                        average_with_simd::<32>(&tape.negative, x_stride),
                        average_with_simd::<32>(&tape.nonnegative, x_stride),
                    ),
                    _ => (
                        average_with_simd::<64>(&tape.negative, x_stride),
                        average_with_simd::<64>(&tape.nonnegative, x_stride),
                    ),
                };
                Self {
                    x_stride,
                    negative,
                    nonnegative,
                    time: step_index,
                    pixel_policy,
                }
            }
            PixelPolicy::Sampling => {
                let negative = sample_with_iterators(&tape.negative, x_stride);
                let nonnegative = sample_with_iterators(&tape.nonnegative, x_stride);
                Self {
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
        let x_stride = find_x_stride(tape.negative.len(), tape.nonnegative.len(), x_goal as usize);

        // When the sample
        if self.x_stride != x_stride {
            return false;
        }
        self.time = step_index;
        let (part, pixels, part_index) = if previous_tape_index < 0 {
            let part_index = (-previous_tape_index - 1) as u64;
            (&tape.negative, &mut self.negative, part_index)
        } else {
            let part_index = previous_tape_index as u64;
            (&tape.nonnegative, &mut self.nonnegative, part_index)
        };
        let pixel_index = x_stride.divide_into(part_index) as usize;
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
                        .map(|i| u32::from(part[i]))
                        .sum();
                    let mean = x_stride.divide_into(sum * 255) as u8;
                    *pixel = Pixel::from(mean);
                } else {
                    let slice = &part[tape_slice_start..tape_slice_end];

                    *pixel = match x_stride {
                        PowerOfTwo::ONE | PowerOfTwo::TWO | PowerOfTwo::FOUR => {
                            let sum: u32 = slice.iter().map(u32::from).sum();
                            (x_stride.divide_into(sum * 255) as u8).into()
                        }
                        PowerOfTwo::EIGHT => average_chunk_with_simd::<8>(slice, x_stride),
                        PowerOfTwo::SIXTEEN => average_chunk_with_simd::<16>(slice, x_stride),
                        PowerOfTwo::THIRTY_TWO => average_chunk_with_simd::<32>(slice, x_stride),
                        _ => average_chunk_with_simd::<64>(slice, x_stride),
                    };
                }
            }
            PixelPolicy::Sampling => {
                *pixel = Pixel::from(part[x_stride * pixel_index]);
            }
        }
        true
    }
}
