use crate::{Pixel, PowerOfTwo, Symbol, ALIGN};
use aligned_vec::AVec;
use core::num::NonZeroU8;
#[cfg(feature = "simd")]
use core::simd::{LaneCount, SupportedLaneCount, prelude::*};
#[cfg(feature = "simd")]
use zerocopy::IntoBytes;

// This is complicated because the tape is in two parts and we always start at 0.
#[inline]
#[must_use]
pub fn find_x_stride(tape_neg_len: usize, tape_non_neg_len: usize, x_goal: usize) -> PowerOfTwo {
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

    let lanes = PowerOfTwo::from_exp(LANES.trailing_zeros() as u8);
    let slice_u8 = values.as_slice().as_bytes();
    let (prefix, chunks, _suffix) = slice_u8.as_simd::<LANES>();

    debug_assert!(prefix.is_empty(), "Expected empty prefix due to alignment");
    let lanes_per_chunk = step.saturating_div(lanes);

    let select_vec = Simd::splat(select.get());

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
    let lanes = PowerOfTwo::from_exp(LANES.trailing_zeros() as u8);
    let lanes_per_chunk = step.saturating_div(lanes);
    debug_assert!(step.divides_usize(chunk.len()));

    let select_vec = Simd::splat(select.get());

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

    (x_stride.divide_into(sum * 255) as u8).into()
}

