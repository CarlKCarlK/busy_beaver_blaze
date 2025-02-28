#![feature(portable_simd)]
use aligned_vec::AVec;
use busy_beaver_blaze::PowerOfTwo;
use core::simd::prelude::*;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn average_with_iterators(a: &AVec<u8>, step: PowerOfTwo) -> AVec<u8> {
    debug_assert!(step.rem_into(a.len() as u64) == 0);
    let mut result = AVec::with_capacity(64, step.divide_into(a.len()));
    for chunk in a.chunks_exact(step.as_usize()) {
        let sum: u32 = chunk.iter().map(|&x| x as u32).sum();
        let average = step.divide_into(sum * 255) as u8;
        result.push(average);
    }
    result
}

fn average_with_simd(a: &AVec<u8>, step: PowerOfTwo) -> AVec<u8> {
    debug_assert!(
        step == PowerOfTwo::THIRTY_TWO,
        "Step must be exactly 32 for SIMD implementation"
    );
    debug_assert!(
        PowerOfTwo::THIRTY_TWO.divides_usize(a.len()),
        "Vector length must be divisible by 32"
    );

    let mut result = AVec::with_capacity(64, step.divide_into(a.len()));

    let (prefix, chunks, suffix) = a.as_slice().as_simd::<32>();

    // Since we're using AVec with 64-byte alignment, the prefix should be empty
    debug_assert!(prefix.is_empty(), "Expected empty prefix due to alignment");

    // Process SIMD chunks directly (each chunk is 32 elements)
    for chunk in chunks {
        // Sum the values within the vector - values are just 0 or 1
        let sum = chunk.reduce_sum();

        // Calculate average (sum * 255 / step)
        let average = step.divide_into(sum as u32 * 255) as u8;
        result.push(average);
    }

    // Process any remaining elements in the suffix
    for chunk in suffix.chunks_exact(step.as_usize()) {
        let sum: u32 = chunk.iter().map(|&x| x as u32).sum();
        let average = step.divide_into(sum * 255) as u8;
        result.push(average);
    }

    result
}

// benchmark average_with_iterators on a random 40K aligned vec using a step size of 32
// Values are either 0 or 1.
fn benchmark_function(c: &mut Criterion) {
    let len = 40_000;
    let step = PowerOfTwo::from_usize(32);
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let a = AVec::from_iter(64, (0..len).map(|_| rng.random::<bool>() as u8));

    let mut group = c.benchmark_group("averaging");

    group.bench_function("iterators", |b| {
        b.iter_with_setup(
            || a.clone(),
            |a_clone| average_with_iterators(black_box(&a_clone), black_box(step)),
        );
    });

    group.bench_function("simd", |b| {
        b.iter_with_setup(
            || a.clone(),
            |a_clone| average_with_simd(black_box(&a_clone), black_box(step)),
        );
    });

    group.finish();
}

criterion_group!(benches, benchmark_function);
criterion_main!(benches);
