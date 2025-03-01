#![feature(portable_simd)]
use aligned_vec::AVec;
use busy_beaver_blaze::{
    ALIGN, PowerOfTwo, average_with_iterators, average_with_simd, average_with_simd_rayon,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};

// benchmark average_with_iterators on a random 40K aligned vec using a step size of 32
// Values are either 0 or 1.
fn small(c: &mut Criterion) {
    let len = 10_048;
    let step = PowerOfTwo::from_usize(64);
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let values = AVec::from_iter(ALIGN, (0..len).map(|_| rng.random::<bool>() as u8));

    let mut group = c.benchmark_group("small");

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

fn large(c: &mut Criterion) {
    let len = 250_000;
    let step = PowerOfTwo::from_usize(512);
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let values = AVec::from_iter(ALIGN, (0..len).map(|_| rng.random::<bool>() as u8));

    let mut group = c.benchmark_group("large");

    group.bench_function("iterators", |b| {
        b.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_iterators(black_box(&values_clone), black_box(step)),
        );
    });

    group.bench_function("simd32", |b| {
        b.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_simd::<32>(black_box(&values_clone), black_box(step)),
        );
    });

    group.bench_function("simd64", |b| {
        b.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_simd::<64>(black_box(&values_clone), black_box(step)),
        );
    });

    group.bench_function("simd32rayon", |b| {
        b.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_simd_rayon::<32>(black_box(&values_clone), black_box(step)),
        );
    });

    group.bench_function("simd64rayon", |b| {
        b.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_simd_rayon::<64>(black_box(&values_clone), black_box(step)),
        );
    });

    group.finish();
}

criterion_group!(benches, small, large);
criterion_main!(benches);
