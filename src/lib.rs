#![feature(portable_simd)]

// cmk00 make custom size details reflected in the URL
// cmk00 Test 9 symbols
// cmk00 generalize parser to support multi-numeral symbols and multi-letter states
// cmk00 should we support L/R/S

// Add the tests module
#[cfg(test)]
mod tests;

// Add modules
mod log_step_iterator;
mod machine;
mod message0;
mod pixel;
mod pixel_policy;
mod png_data_iterator;
mod png_data_layers;
mod power_of_two;
mod snapshot;
mod space_by_time;
mod space_by_time_machine;
mod space_time_layers;
mod spaceline;
mod spacelines;
mod symbol;
mod tape;

use aligned_vec::AVec;
use core::num::NonZeroU8;
use core::panic;
use core::simd::{LaneCount, SupportedLaneCount, prelude::*};
use derive_more::{Error as DeriveError, derive::Display};
use png::{BitDepth, ColorType, Encoder};
use snapshot::Snapshot;
use symbol::Symbol;
use thousands::Separable;
use zerocopy::IntoBytes;
// Export types from modules
pub use log_step_iterator::LogStepIterator;
pub use machine::Machine;
pub use pixel::Pixel;
pub use pixel_policy::PixelPolicy;
pub use png_data_iterator::PngDataIterator;
pub use png_data_layers::PngDataLayers;
pub use power_of_two::PowerOfTwo;
pub use space_by_time::SpaceByTime;
pub use space_by_time_machine::SpaceByTimeMachine;
pub use spaceline::Spaceline;
pub use tape::Tape;

pub const ALIGN: usize = 64;

pub const BB2_CHAMP: &str = "
	A	B
0	1RB	1LA
1	1LB	1RH
";

pub const BB3_CHAMP: &str = "
	A	B	C
0	1RB	0RC	1LC
1	1RH	1RB	1LA
";
pub const BB4_CHAMP: &str = "
	A	B	C	D
0	1RB	1LA	1RH	1RD
1	1LB	0LC	1LD	0RA
";

pub const BB5_CHAMP: &str = "
    A	B	C	D	E
0	1RB	1RC	1RD	1LA	1RH
1	1LC	1RB	0LE	1LD	0LA
";

pub const BB5_CHAMP_SHIFT_2: &str = "
    A	B	C	D	E
0	2RB	2RC	2RD	2LA	2RH
1	--- --- --- --- ---
2	2LC	2RB	0LE	2LD	0LA
";

pub const BB6_CONTENDER: &str = "
    	A	B	C	D	E	F
0	1RB	1RC	1LC	0LE	1LF	0RC
1	0LD	0RF	1LA	1RH	0RB	0RE
";

pub const BB6_CONTENDER_SHIFT2: &str = "
    	A	B	C	D	E	F
0	2RB	2RC	2LC	0LE	2LF	0RC
1	--- --- --- --- --- ---
2	0LD	0RF	2LA	2RH	0RB	0RE
";

pub const MACHINE_7_135_505_A: &str = "   
0	1
A	1RB	0LD
B	1RC	---
C	1LD	1RA
D	1RE	1LC
E	0LA	0RE
";
pub const MACHINE_7_135_505_B: &str = "1RB0LD_1RC---_1LD1RA_1RE1LC_0LA0RE";

// https://github.com/sligocki/busy-beaver/blob/main/Machines/bb/3x3
pub const BB_3_3_355317: &str = "1RB2LA1RA_1LA1RZ1RC_2RB1RC2RB";
pub const BIGFOOT33: &str = "1RB2RA1LC_2LC1RB2RB_---2LA1LA";
pub const BIGFOOT72: &str = "0RB1RB_1LC0RA_1RE1LF_1LF1RE_0RD1RD_1LG0LG_---1LB";
pub const BRADY: &str = "1RB2LB1LC_1LA2RB1RB_1RZ2LA0LC";
pub const BB_2_5_CHAMP_AUG25: &str = "1RB3LA4RB0RB2LA_1LB2LA3LA1RA1RZ";

pub const DEFAULT_COLORS: &[[u8; 3]] = &[
    [255, 255, 255], // white
    [255, 165, 0],   // orange
    [255, 255, 0],   // yellow
    [255, 0, 255],   // magenta
    [0, 255, 255],   // cyan
    [0, 128, 0],     // green
    [0, 0, 255],     // blue
    [75, 0, 130],    // indigo
    [238, 130, 238], // violet
    [255, 0, 0],     // red
    [0, 0, 0],       // black
];

/// Normalize a color palette for a given symbol count, cycling as needed.
/// Returns an error if fewer than 2 colors are provided.
pub fn normalize_colors(input: &[[u8; 3]], symbol_count: usize) -> Result<Vec<[u8; 3]>, Error> {
    let colors = if input.is_empty() {
        DEFAULT_COLORS
    } else {
        input
    };
    if colors.len() < 2 {
        return Err(Error::UnexpectedColorCount { got: colors.len() });
    }
    Ok(core::iter::once(colors[0])
        .chain(colors[1..].iter().copied().cycle())
        .take(symbol_count)
        .collect())
}

/// A trait for iterators that can print debug output at intervals.
pub trait DebuggableIterator: Iterator {
    /// Runs the iterator while printing debug output at intervals.
    #[inline]
    fn debug_count(&mut self, debug_interval: usize) -> usize
    where
        Self: Sized + core::fmt::Debug, // ✅ Ensure Debug is implemented
    {
        let mut step_index = 0;
        let mut countdown = debug_interval; // New countdown variable

        println!("Step {}: {:?}", step_index.separate_with_commas(), self);

        while self.next().is_some() {
            step_index += 1;
            countdown -= 1;

            if countdown == 0 {
                println!("Step {}: {:?}", step_index.separate_with_commas(), self);
                countdown = debug_interval; // Reset countdown
            }
        }

        step_index + 1 // Convert last index into count
    }
}

// Implement the trait for all Iterators
#[allow(clippy::missing_trait_methods)]
impl<T> DebuggableIterator for T where T: Iterator + core::fmt::Debug {}

/// Error type for parsing a `Program` from a string.
#[derive(Debug, Display, DeriveError)]
pub enum Error {
    #[display("Invalid number format: {}", _0)]
    ParseIntError(core::num::ParseIntError),

    #[display("Invalid character encountered in part")]
    InvalidChar { got: char },

    #[display("Unexpected empty field in input")]
    MissingField,

    #[display("Unexpected symbols count. and got {}", got)]
    UnexpectedSymbolCount { got: usize },

    #[display("Unexpected colors count. Got {}", got)]
    UnexpectedColorCount { got: usize },

    #[display("Unexpected states count. Expected {} and got {}", expected, got)]
    InvalidStatesCount { expected: usize, got: usize },

    #[display(
        "Image data layers with bad pixel count. Expected {} and got {}",
        expected,
        got
    )]
    ImageDataWithBadPixelCount { expected: usize, got: usize },

    #[display("Unexpected symbol encountered")]
    UnexpectedSymbol,

    #[display("Unexpected state encountered")]
    UnexpectedState,

    #[display("Invalid encoding encountered")]
    EncodingError,

    #[display("Image data layers are empty")]
    EmptyImageDataLayers,

    #[display("Unknown program format")]
    UnknownProgramFormat,
}

// Implement conversions manually where needed
impl From<core::num::ParseIntError> for Error {
    fn from(err: core::num::ParseIntError) -> Self {
        Self::ParseIntError(err)
    }
}

// This is complicated because the tape is in two parts and we always start at 0.
fn find_x_stride(tape_neg_len: usize, tape_non_neg_len: usize, x_goal: usize) -> PowerOfTwo {
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

#[inline]
#[must_use]
#[allow(clippy::integer_division_remainder_used)]
pub const fn find_y_stride(len: u64, y_goal: u32) -> PowerOfTwo {
    let threshold = 2 * y_goal as u64;
    // Compute the ceiling of (len + 1) / threshold.
    // Note: (len + threshold) / threshold is equivalent to ceil((len + 1) / threshold)
    let ceiling_ratio = (len + threshold) / threshold;
    let floor_log2 = 63 - ceiling_ratio.leading_zeros();
    let exponent = floor_log2 + ((!ceiling_ratio.is_power_of_two()) as u32);

    PowerOfTwo::from_exp(exponent as u8)
}

// cmk0 later consider mixing with gamma
// cmk0 consider special cases 2 symbols (1 layer) and 3 symbols (2 layers)
//       with a palette.
#[allow(clippy::integer_division_remainder_used)]
fn encode_png(
    width: u32,
    height: u32,
    colors: &[[u8; 3]],
    image_data_layers: &[AVec<u8>],
) -> Result<Vec<u8>, Error> {
    let mut buf = Vec::new();
    {
        let pixel_count = (width as usize) * (height as usize);
        if image_data_layers.is_empty() {
            return Err(Error::EmptyImageDataLayers);
        }
        for image_data in image_data_layers {
            if image_data.len() != pixel_count {
                return Err(Error::ImageDataWithBadPixelCount {
                    expected: pixel_count,
                    got: image_data.len(),
                });
            }
        }

        let colors = normalize_colors(colors, image_data_layers.len() + 1)?;

        let mut image_data = Vec::with_capacity(pixel_count * 3);

        for pixel_index in 0..pixel_count {
            // Accumulator for RGB channels; use wider type to avoid wrap. u16
            // would likely be enough but there would be a chance of rounding error
            // causing overflow.
            let mut channels = [0u32; 3];
            // Remaining weight for the base color
            let mut weight0 = 255u8;
            for (layer_data, &color) in image_data_layers.iter().zip(&colors[1..]) {
                let weight_nonzero = layer_data[pixel_index];
                weight0 = weight0.saturating_sub(weight_nonzero);
                for (acc, ch) in channels.iter_mut().zip(color) {
                    *acc += (ch as u32) * (weight_nonzero as u32);
                }
            }

            // Base color soaks up the remainder (branchless)
            for (acc, ch0) in channels.iter_mut().zip(colors[0]) {
                *acc += (ch0 as u32) * (weight0 as u32);
            }

            // Divide once per channel and clamp to u8
            let out = channels.map(|x| ((x / 255).min(255)) as u8);
            image_data.extend_from_slice(&out);
        }

        let mut encoder = Encoder::new(&mut buf, width, height);
        encoder.set_color(ColorType::Rgb);
        encoder.set_depth(BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|_| Error::EncodingError)?;
        writer
            .write_image_data(image_data.as_ref())
            .map_err(|_| Error::EncodingError)?;
    };
    Ok(buf)
}

#[must_use]
pub fn average_with_iterators(
    select: NonZeroU8,
    values: &AVec<Symbol>,
    step: PowerOfTwo,
) -> AVec<Pixel> {
    let mut result: AVec<Pixel, _> = AVec::with_capacity(ALIGN, step.div_ceil_into(values.len()));

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
    let mut result: AVec<Pixel, _> = AVec::with_capacity(ALIGN, step.div_ceil_into(values.len()));

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

// TODO move this to tape and give a better name
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

    // ✅ Pre-fill `result` with zeros to avoid `push()`
    let mut result: AVec<Pixel, _> = AVec::with_capacity(ALIGN, capacity);
    result.resize(capacity, Pixel::WHITE);

    let lanes = PowerOfTwo::from_exp(LANES.trailing_zeros() as u8);
    let slice_u8 = values.as_slice().as_bytes();
    let (prefix, chunks, _suffix) = slice_u8.as_simd::<LANES>();

    debug_assert!(prefix.is_empty(), "Expected empty prefix due to alignment");
    let lanes_per_chunk = step.saturating_div(lanes);

    let select_vec = Simd::splat(select.get());

    // ✅ Process chunks using `zip()`, no `push()`
    if lanes_per_chunk == PowerOfTwo::ONE {
        for (average, chunk) in result.iter_mut().zip(chunks.iter()) {
            let sum = chunk.simd_eq(select_vec).to_bitmask().count_ones();
            *average = (step.divide_into(sum * 255) as u8).into();
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

    // ✅ Efficiently handle remaining elements without `push()`
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
    // convert to simd
    let (prefix, sub_chunks, suffix) = chunk.as_bytes().as_simd::<LANES>();

    debug_assert!(
        prefix.is_empty() && suffix.is_empty(),
        "Expected empty prefix due to alignment"
    );
    let lanes = PowerOfTwo::from_exp(LANES.trailing_zeros() as u8);
    let lanes_per_chunk = step.saturating_div(lanes);
    debug_assert!(step.divides_usize(chunk.len()));

    let select_vec = Simd::splat(select.get());

    // ✅ Process chunks using `zip()`, no `push()`
    if lanes_per_chunk == PowerOfTwo::ONE {
        debug_assert!(sub_chunks.len() == 1, "Expected one chunk");
        let sum = sub_chunks[0].simd_eq(select_vec).to_bitmask().count_ones();
        (step.divide_into(sum * 255) as u8).into()
    } else {
        let sum: u32 = sub_chunks
            .iter()
            .map(|sub_chunk| sub_chunk.simd_eq(select_vec).to_bitmask().count_ones())
            .sum();
        (step.divide_into(sum * 255) as u8).into()
    }
}

#[inline]
pub fn is_even<T>(x: T) -> bool
where
    T: Copy + core::ops::BitAnd<Output = T> + core::ops::Sub<Output = T> + From<u8> + PartialEq,
{
    (x & T::from(1)) == T::from(0)
}

#[inline]
/// This returns the largest power of two that is less than or equal
/// to the input number x.
#[must_use]
pub const fn prev_power_of_two(x: usize) -> usize {
    debug_assert!(x > 0, "x must be greater than 0");
    1usize << (usize::BITS as usize - x.leading_zeros() as usize - 1)
}

fn compress_packed_data_if_one_too_big(
    mut packed_data: AVec<u8>,
    pixel_policy: PixelPolicy,
    y_goal: u32,
    x_actual: u32,
    y_actual: u32,
) -> (AVec<u8>, u32) {
    if y_actual < 2 * y_goal {
        (packed_data, y_actual)
    } else {
        assert!(y_actual == 2 * y_goal, "y_actual must be 2 * y_goal");
        // reduce the # of rows in half my averaging
        let mut new_packed_data = AVec::with_capacity(ALIGN, x_actual as usize * y_goal as usize);
        new_packed_data.resize(x_actual as usize * y_goal as usize, 0u8);

        packed_data
            .chunks_exact_mut(x_actual as usize * 2)
            .zip(new_packed_data.chunks_exact_mut(x_actual as usize))
            .for_each(|(chunk, new_chunk)| {
                let (left, right) = chunk.split_at_mut(x_actual as usize);
                // TODO why binning in the inner loop?
                match pixel_policy {
                    // Can't use simd because chunks left & right may not be aligned.
                    PixelPolicy::Binning => Pixel::slice_merge_bytes_no_simd(left, right),
                    PixelPolicy::Sampling => (),
                }

                // by design new_chunk is the same size as left, so copy the bytes from left to new_chunk
                new_chunk.copy_from_slice(left);
            });
        (new_packed_data, y_goal)
    }
}

pub mod test_utils {
    use crate::{Pixel, is_even};
    use aligned_vec::AVec;

    pub fn compress_x_no_simd_binning(pixels: &mut AVec<Pixel>) {
        let len = pixels.len();
        let mut write_index = 0;

        let mut i = 0;
        while i + 1 < len {
            pixels[write_index] = pixels[i] + pixels[i + 1]; // Overlapping write
            write_index += 1;
            i += 2;
        }
        if !is_even(len) {
            pixels[write_index] = pixels[len - 1] + Pixel::WHITE; // Handle last odd element
            write_index += 1;
        }

        pixels.truncate(write_index);
    }
}
