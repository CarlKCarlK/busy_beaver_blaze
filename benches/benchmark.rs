#![feature(portable_simd)]
#![feature(slice_take)]
use aligned_vec::AVec;
use busy_beaver_blaze::PowerOfTwo;
use core::simd::prelude::*;
use core::simd::{LaneCount, SupportedLaneCount};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

const ALIGN: usize = 64;

fn average_with_iterators(values: &AVec<u8>, step: PowerOfTwo) -> AVec<u8> {
    let mut result = AVec::with_capacity(ALIGN, step.div_ceil_into(values.len()));

    // Process complete chunks
    let chunk_iter = values.chunks_exact(step.as_usize());
    let remainder = chunk_iter.remainder();

    for chunk in chunk_iter {
        let sum: u32 = chunk.iter().map(|&x| x as u32).sum();
        let average = step.divide_into(sum * 255) as u8;
        result.push(average);
    }

    // Handle the remainder - pad with zeros
    if !remainder.is_empty() {
        let sum: u32 = remainder.iter().map(|&x| x as u32).sum();
        // We need to divide by step size, not remainder.len()
        let average = step.divide_into(sum * 255) as u8;
        result.push(average);
    }

    result
}

fn average_with_simd<const LANES: usize>(values: &AVec<u8>, step: PowerOfTwo) -> AVec<u8>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    debug_assert!(
        { LANES } <= step.as_usize() && { LANES } <= { ALIGN },
        "LANES must be less than or equal to step and alignment"
    );

    let mut result = AVec::with_capacity(ALIGN, step.div_ceil_into(values.len()));
    let lanes = PowerOfTwo::from_exp(LANES.trailing_zeros() as u8);

    let (prefix, chunks, _suffix) = values.as_slice().as_simd::<LANES>();

    // Since we're using AVec with 64-byte alignment, the prefix should be empty
    debug_assert!(prefix.is_empty(), "Expected empty prefix due to alignment");
    // Process SIMD chunks directly (each chunk is N elements)
    let lanes_per_chunk = step.saturating_div(lanes);

    if lanes_per_chunk == PowerOfTwo::ONE {
        for chunk in chunks {
            let sum = chunk.reduce_sum() as u32;
            let average = step.divide_into(sum * 255) as u8;
            result.push(average);
        }
    } else {
        let mut chunk_iter = chunks.chunks_exact(step.saturating_div(lanes).as_usize());

        // Process complete chunks
        for sub_chunk in &mut chunk_iter {
            // Sum the values within the vector - values are just 0 or 1
            let sum: u32 = sub_chunk
                .iter()
                .map(|chunk| chunk.reduce_sum() as u32)
                .sum();
            let average = step.divide_into(sum * 255) as u8;
            result.push(average);
        }
    }

    // How many elements would we need to make values.len() a multiple of step?
    let missing_items = step.offset_to_align(values.len());
    if missing_items > 0 {
        // sum the last missing_items
        let sum: u32 = values
            .iter()
            .rev()
            .take(missing_items)
            .map(|&x| x as u32)
            .sum();
        let average = step.divide_into(sum * 255) as u8;
        result.push(average);
    }

    result
}

// benchmark average_with_iterators on a random 40K aligned vec using a step size of 32
// Values are either 0 or 1.
fn benchmark_function(c: &mut Criterion) {
    let len = 10_048;
    let step = PowerOfTwo::from_usize(64);
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let values = AVec::from_iter(64, (0..len).map(|_| rng.random::<bool>() as u8));

    let mut group = c.benchmark_group("averaging");

    group.bench_function("iterators", |b| {
        b.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_iterators(black_box(&values_clone), black_box(step)),
        );
    });

    group.bench_function("simd", |b| {
        b.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_simd::<32>(black_box(&values_clone), black_box(step)),
        );
    });

    group.finish();
}

criterion_group!(benches, benchmark_function);
criterion_main!(benches);
