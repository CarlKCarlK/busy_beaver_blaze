#![feature(portable_simd)]

// Add the tests module
#[cfg(test)]
mod tests;

// Add modules
mod bool_u8;
mod log_step_iterator;
mod machine;
mod pixel;
mod pixel_policy;
mod power_of_two;
mod space_by_time;
mod space_by_time_machine;
mod spaceline;
mod spacelines;
mod tape;

// cmk00 Ideas for speedup:
// cmk00    Use nightly simd to average adjacent cells (only useful when at higher smoothing)
// cmk00    Build up 64 (or 128 or 256) rows without merging then use a Rayon parallel tree merge (see https://chatgpt.com/share/67bb94cb-4ba4-800c-b430-c45a5eb46715)
// cmk00    Better than doing a tree merge would be having different final lines processed in parallel

// cmk Could u8 in tape be a bool? (also in average_with_X)

use aligned_vec::AVec;
use bool_u8::BoolU8;
use core::simd::{LaneCount, SupportedLaneCount, prelude::*};
use derive_more::{Error as DeriveError, derive::Display};
use pixel::Pixel;
use png::{BitDepth, ColorType, Encoder};
use thousands::Separable;
use zerocopy::IntoBytes;
// Export types from modules
pub use log_step_iterator::LogStepIterator;
pub use machine::Machine;
pub use pixel_policy::PixelPolicy;
pub use power_of_two::PowerOfTwo;
pub use space_by_time::SpaceByTime;
pub use space_by_time_machine::SpaceByTimeMachine;
pub use tape::Tape;

const LANES_CMK: usize = 32;
pub const ALIGN: usize = 64;

// use web_sys::console;

// cmk is the image size is a power of 2, then don't apply filters (may be a bad idea, because user doesn't control native size exactly)
// cmk0 see if can remove more as_u64()'s

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

pub const BB6_CONTENDER: &str = "
    	A	B	C	D	E	F
0	1RB	1RC	1LC	0LE	1LF	0RC
1	0LD	0RF	1LA	1RH	0RB	0RE
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
    InvalidChar,

    #[display("Unexpected empty field in input")]
    MissingField,

    #[display("Unexpected symbols count. Expected {} and got {}", expected, got)]
    InvalidSymbolsCount { expected: usize, got: usize },

    #[display("Unexpected states count. Expected {} and got {}", expected, got)]
    InvalidStatesCount { expected: usize, got: usize },

    #[display("Failed to convert to array")]
    ArrayConversionError,

    #[display("Unexpected symbol encountered")]
    UnexpectedSymbol,

    #[display("Unexpected state encountered")]
    UnexpectedState,

    #[display("Invalid encoding encountered")]
    EncodingError,

    #[display("Unexpected format")]
    UnexpectedFormat,
}

// Implement conversions manually where needed
impl From<core::num::ParseIntError> for Error {
    fn from(err: core::num::ParseIntError) -> Self {
        Self::ParseIntError(err)
    }
}

fn sample_rate(row: u64, goal: u32) -> PowerOfTwo {
    let threshold = 2 * goal;
    let ratio = (row + 1) as f64 / threshold as f64;

    // For rows < threshold, ratio < 1, log2(ratio) is negative, and the ceil clamps to 0.
    let exponent = ratio.log2().ceil().max(0.0) as u8;
    PowerOfTwo::from_exp(exponent)
}

#[allow(clippy::integer_division_remainder_used)]
fn encode_png(width: u32, height: u32, image_data: &[u8]) -> Result<Vec<u8>, Error> {
    let mut buf = Vec::new();
    {
        if image_data.len() != (width * height) as usize {
            return Err(Error::EncodingError);
        }
        let mut encoder = Encoder::new(&mut buf, width, height);
        encoder.set_color(ColorType::Indexed);
        encoder.set_depth(BitDepth::Eight);

        // Generate a palette with 256 shades from white (255,255,255) to bright orange (255,165,0)
        let mut palette = Vec::with_capacity(256 * 3);
        for i in 0u16..256 {
            let green = 255 - ((255 - 165) * i / 255); // Green fades from 255 to 165
            let blue = 255 - (255 * i / 255); // Blue fades from 255 to 0
            palette.extend_from_slice(&[255, green as u8, blue as u8]);
        }

        // Set the palette before writing the header
        encoder.set_palette(palette);

        let mut writer = encoder.write_header().map_err(|_| Error::EncodingError)?;
        writer
            .write_image_data(image_data)
            .map_err(|_| Error::EncodingError)?;
    };
    Ok(buf)
}
#[must_use]
pub fn average_with_iterators(values: &AVec<BoolU8>, step: PowerOfTwo) -> AVec<Pixel> {
    let mut result: AVec<Pixel, _> = AVec::with_capacity(ALIGN, step.div_ceil_into(values.len()));

    // Process complete chunks
    let chunk_iter = values.chunks_exact(step.as_usize());
    let remainder = chunk_iter.remainder();

    for chunk in chunk_iter {
        let sum: u32 = chunk.iter().map(u32::from).sum();
        let average = step.divide_into(sum * 255).into();
        result.push(average);
    }

    // Handle the remainder - pad with zeros
    if !remainder.is_empty() {
        let sum: u32 = remainder.iter().map(u32::from).sum();
        // We need to divide by step size, not remainder.len()
        let average = step.divide_into(sum * 255).into();
        result.push(average);
    }

    result
}

#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn average_with_simd<const LANES: usize>(values: &AVec<BoolU8>, step: PowerOfTwo) -> AVec<Pixel>
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

    // ✅ Process chunks using `zip()`, no `push()`
    if lanes_per_chunk == PowerOfTwo::ONE {
        for (average, chunk) in result.iter_mut().zip(chunks.iter()) {
            let sum = chunk.reduce_sum() as u32;
            *average = (step.divide_into(sum * 255) as u8).into();
        }
    } else {
        let mut chunk_iter = chunks.chunks_exact(lanes_per_chunk.as_usize());

        for (average, sub_chunk) in result.iter_mut().zip(&mut chunk_iter) {
            let sum: u32 = sub_chunk
                .iter()
                .map(|chunk| chunk.reduce_sum() as u32)
                .sum();
            *average = step.divide_into(sum * 255).into();
        }
    }

    // ✅ Efficiently handle remaining elements without `push()`
    let unused_items = step.rem_into_usize(values_len);
    if unused_items > 0 {
        let sum: u32 = values[values_len - unused_items..]
            .iter()
            .map(u32::from)
            .sum();
        if let Some(last) = result.last_mut() {
            *last = step.divide_into(sum * 255).into();
        }
    }

    result
}

#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn average_chunk_with_simd<const LANES: usize>(chunk: &[BoolU8], step: PowerOfTwo) -> Pixel
where
    LaneCount<LANES>: SupportedLaneCount,
{
    // cmk00000 make these debug_asserts
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

    // ✅ Process chunks using `zip()`, no `push()`
    if lanes_per_chunk == PowerOfTwo::ONE {
        debug_assert!(sub_chunks.len() == 1, "Expected one chunk");
        let sum = sub_chunks[0].reduce_sum() as u32;
        (step.divide_into(sum * 255) as u8).into()
    } else {
        let sum: u32 = sub_chunks
            .iter()
            .map(|sub_chunk| sub_chunk.reduce_sum() as u32)
            .sum();
        (step.divide_into(sum * 255) as u8).into()
    }
}

#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn average_with_simd_push<const LANES: usize>(
    values: &AVec<BoolU8>,
    step: PowerOfTwo,
) -> AVec<Pixel>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    assert!(
        { LANES } <= step.as_usize() && { LANES } <= { ALIGN },
        "LANES must be less than or equal to step and alignment"
    );
    let values_u8 = values.as_bytes();
    let values_len = values_u8.len();
    let mut result: AVec<Pixel, _> = AVec::with_capacity(ALIGN, step.div_ceil_into(values_len));
    let lanes = PowerOfTwo::from_exp(LANES.trailing_zeros() as u8);

    let (prefix, chunks, _suffix) = values_u8.as_simd::<LANES>();

    // Since we're using AVec with 64-byte alignment, the prefix should be empty
    debug_assert!(prefix.is_empty(), "Expected empty prefix due to alignment");
    // Process SIMD chunks directly (each chunk is N elements)
    let lanes_per_chunk = step.saturating_div(lanes);

    if lanes_per_chunk == PowerOfTwo::ONE {
        for chunk in chunks {
            let sum = chunk.reduce_sum() as u32;
            let average = step.divide_into(sum * 255) as u8;
            result.push(average.into());
        }
    } else {
        let mut chunk_iter = chunks.chunks_exact(lanes_per_chunk.as_usize());

        // Process complete chunks
        for sub_chunk in &mut chunk_iter {
            // Sum the values within the vector - values are just 0 or 1
            let sum: u32 = sub_chunk
                .iter()
                .map(|chunk| chunk.reduce_sum() as u32)
                .sum();
            let average = step.divide_into(sum * 255) as u8;
            result.push(average.into());
        }
    }

    // How many elements are unprocessed?
    let unused_items = step.rem_into_usize(values_len);
    if unused_items > 0 {
        // sum the last missing_items
        let sum: u32 = values_u8
            .iter()
            .rev()
            .take(unused_items)
            .map(|&x| x as u32)
            .sum();
        let average = step.divide_into(sum * 255) as u8;
        result.push(average.into());
    }

    result
}

#[allow(clippy::missing_panics_doc, clippy::shadow_reuse)]
#[must_use]
pub fn average_with_simd_count_ones64(values: &AVec<BoolU8>, step: PowerOfTwo) -> AVec<Pixel> {
    const LANES: usize = 64;
    assert!(
        { LANES } <= step.as_usize() && { LANES } <= { ALIGN },
        "LANES must be less than or equal to step and alignment"
    );

    let values_len = values.len();
    let mut result: AVec<Pixel, _> = AVec::with_capacity(ALIGN, step.div_ceil_into(values_len));
    let lanes = PowerOfTwo::from_exp(LANES.trailing_zeros() as u8);

    let (prefix, chunks, _suffix) = values.as_bytes().as_simd::<LANES>();

    // Since we're using AVec with 64-byte alignment, the prefix should be empty
    debug_assert!(prefix.is_empty(), "Expected empty prefix due to alignment");
    // Process SIMD chunks directly (each chunk is N elements)
    let lanes_per_chunk = step.saturating_div(lanes);

    if lanes_per_chunk == PowerOfTwo::ONE {
        for chunk in chunks {
            // reinterpret as &simd<u64,1> in place using raw pointers instead of transmute
            let chunk = unsafe {
                &*core::ptr::from_ref::<core::simd::Simd<u8, 64>>(chunk)
                    .cast::<core::simd::Simd<u64, 8>>()
            };
            let sum = chunk.count_ones().reduce_sum() as u32;
            let average = step.divide_into(sum * 255) as u8;
            result.push(average.into());
        }
    } else {
        let mut chunk_iter = chunks.chunks_exact(lanes_per_chunk.as_usize());

        // Process complete chunks
        for sub_chunk in &mut chunk_iter {
            // Sum the values within the vector - values are just 0 or 1
            let sum: u32 = sub_chunk
                .iter()
                .map(|chunk| {
                    let chunk = unsafe {
                        &*core::ptr::from_ref::<core::simd::Simd<u8, 64>>(chunk)
                            .cast::<core::simd::Simd<u64, 8>>()
                    };
                    chunk.count_ones().reduce_sum() as u32
                })
                .sum();
            let average = step.divide_into(sum * 255) as u8;
            result.push(average.into());
        }
    }

    // How many elements are unprocessed?
    let unused_items = step.rem_into_usize(values_len);
    if unused_items > 0 {
        // sum the last missing_items
        let sum: u32 = values.iter().rev().take(unused_items).map(u32::from).sum();
        let average = step.divide_into(sum * 255) as u8;
        result.push(average.into());
    }

    result
}

#[inline]
fn fast_is_even<T>(x: T) -> bool
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
