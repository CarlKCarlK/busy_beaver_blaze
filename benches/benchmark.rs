#![cfg(not(target_arch = "wasm32"))]
#![feature(portable_simd)]

use aligned_vec::AVec;
use busy_beaver_blaze::test_utils::compress_x_no_simd_binning;
use busy_beaver_blaze::{
    ALIGN, Pixel, PowerOfTwo, Spaceline, average_with_iterators, average_with_simd,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};

// benchmark average_with_iterators on a random 40K aligned vec using a step size of 32
// Values are either 0 or 1.
fn small(criterion: &mut Criterion) {
    let len = 10_048;
    let step = PowerOfTwo::from_usize_unchecked(64);
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let values = AVec::from_iter(ALIGN, (0..len).map(|_| rng.random::<bool>().into()));

    let mut group = criterion.benchmark_group("small");

    group.bench_function("iterators", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_iterators(black_box(&values_clone), black_box(step)),
        );
    });

    group.finish();
}

fn large(criterion: &mut Criterion) {
    let len = 250_000;
    let step = PowerOfTwo::from_usize_unchecked(512);
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let values = AVec::from_iter(ALIGN, (0..len).map(|_| rng.random::<bool>().into()));

    let mut group = criterion.benchmark_group("large");

    group.bench_function("iterators", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_iterators(black_box(&values_clone), black_box(step)),
        );
    });

    group.bench_function("simd32", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_simd::<32>(black_box(&values_clone), black_box(step)),
        );
    });

    group.bench_function("simd64", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_simd::<64>(black_box(&values_clone), black_box(step)),
        );
    });

    group.finish();
}

fn len_100m(criterion: &mut Criterion) {
    let len = 100_000_000;
    let step = PowerOfTwo::from_usize(65536);
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let values = AVec::from_iter(ALIGN, (0..len).map(|_| rng.random::<bool>().into()));

    let mut group = criterion.benchmark_group("len_100m");

    group.bench_function("iterators", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_iterators(black_box(&values_clone), black_box(step)),
        );
    });

    group.bench_function("simd32", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_simd::<32>(black_box(&values_clone), black_box(step)),
        );
    });

    group.bench_function("simd64", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_simd::<64>(black_box(&values_clone), black_box(step)),
        );
    });

    group.finish();
}

fn resample(criterion: &mut Criterion) {
    let len = 4_000;
    let seed = 1;
    let mut rng = StdRng::seed_from_u64(seed);
    let pixels: AVec<Pixel, _> =
        AVec::from_iter(ALIGN, (0..len).map(|_| rng.random::<bool>().into()));

    let mut group = criterion.benchmark_group("resample");

    group.bench_function("iterators", |bencher| {
        bencher.iter_with_setup(
            || pixels.clone(),
            |mut pixels_clone| compress_x_no_simd_binning(black_box(&mut pixels_clone)),
        );
    });

    group.bench_function("simd", |bencher| {
        bencher.iter_with_setup(
            || pixels.clone(),
            |mut pixels_clone| Spaceline::compress_x_simd_binning(black_box(&mut pixels_clone)),
        );
    });

    group.finish();
}

criterion_group!(benches, small, large, len_100m, resample);
criterion_main!(benches);
