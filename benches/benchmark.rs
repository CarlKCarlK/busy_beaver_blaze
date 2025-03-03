#![feature(portable_simd)]
use aligned_vec::AVec;
use busy_beaver_blaze::{
    ALIGN, PowerOfTwo, average_with_iterators, average_with_simd, average_with_simd_count_ones64,
    average_with_simd_push, average_with_simd_rayon,
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

    group.bench_function("simd", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_simd_push::<32>(black_box(&values_clone), black_box(step)),
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

    // group.bench_function("simd32rayon", |bencher| {
    //     bencher.iter_with_setup(
    //         || values.clone(),
    //         |values_clone| average_with_simd_rayon::<32>(black_box(&values_clone), black_box(step)),
    //     );
    // });

    // group.bench_function("simd64rayon", |bencher| {
    //     bencher.iter_with_setup(
    //         || values.clone(),
    //         |values_clone| average_with_simd_rayon::<64>(black_box(&values_clone), black_box(step)),
    //     );
    // });

    // group.bench_function("simd64_count_ones", |bencher| {
    //     bencher.iter_with_setup(
    //         || values.clone(),
    //         |values_clone| {
    //             average_with_simd_count_ones64(black_box(&values_clone), black_box(step))
    //         },
    //     );
    // });

    group.bench_function("simd64_push", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_simd_push::<64>(black_box(&values_clone), black_box(step)),
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

    group.bench_function("simd32rayon2", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                average_with_simd_rayon::<32>(black_box(&values_clone), black_box(step), 2)
            },
        );
    });

    group.bench_function("simd64rayon1", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                average_with_simd_rayon::<64>(black_box(&values_clone), black_box(step), 1)
            },
        );
    });

    group.bench_function("simd64rayon2", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                average_with_simd_rayon::<64>(black_box(&values_clone), black_box(step), 2)
            },
        );
    });

    group.bench_function("simd64rayon4", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                average_with_simd_rayon::<64>(black_box(&values_clone), black_box(step), 4)
            },
        );
    });
    group.bench_function("simd64rayon8", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                average_with_simd_rayon::<64>(black_box(&values_clone), black_box(step), 8)
            },
        );
    });
    group.bench_function("simd64rayon16", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                average_with_simd_rayon::<64>(black_box(&values_clone), black_box(step), 16)
            },
        );
    });
    group.bench_function("simd64_count_ones", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                average_with_simd_count_ones64(black_box(&values_clone), black_box(step))
            },
        );
    });

    group.bench_function("simd64_push", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| average_with_simd_push::<64>(black_box(&values_clone), black_box(step)),
        );
    });

    group.finish();
}

criterion_group!(benches, small, large, len_100m);
criterion_main!(benches);
