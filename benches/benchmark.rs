#![cfg(not(target_arch = "wasm32"))]

use core::num::NonZeroU8;

use aligned_vec::AVec;
use busy_beaver_blaze::test_utils::compress_x_no_simd_binning;
#[cfg(feature = "gpu")]
use busy_beaver_blaze::{average_with_gpu, GpuAverage};
use busy_beaver_blaze::{Pixel, PowerOfTwo, Spaceline, ALIGN};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};

const SELECT_ONE: NonZeroU8 = NonZeroU8::new(1).unwrap();

// benchmark average_with_iterators on a random 40K aligned vec using a step size of 32
// Values are either 0 or 1.
fn small(criterion: &mut Criterion) {
    let len = 10_048;
    let step = PowerOfTwo::from_usize_unchecked(64);
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let values = AVec::from_iter(ALIGN, (0..len).map(|_| rng.random_range(0u8..=1).into()));

    let mut group = criterion.benchmark_group("small");

    group.bench_function("iterators", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                Spaceline::average_with_iterators(
                    SELECT_ONE,
                    black_box(&values_clone),
                    black_box(step),
                )
            },
        );
    });

    group.finish();
}

fn large(criterion: &mut Criterion) {
    let len = 250_000;
    let step = PowerOfTwo::from_usize_unchecked(512);
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let values = AVec::from_iter(ALIGN, (0..len).map(|_| rng.random_range(0u8..=1).into()));

    let mut group = criterion.benchmark_group("large");

    group.bench_function("iterators", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                Spaceline::average_with_iterators(
                    SELECT_ONE,
                    black_box(&values_clone),
                    black_box(step),
                )
            },
        );
    });

    group.bench_function("simd32", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                Spaceline::average_with_simd::<32>(
                    SELECT_ONE,
                    black_box(&values_clone),
                    black_box(step),
                )
            },
        );
    });

    group.bench_function("simd64", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                Spaceline::average_with_simd::<64>(
                    SELECT_ONE,
                    black_box(&values_clone),
                    black_box(step),
                )
            },
        );
    });

    group.finish();
}

fn len_100m(criterion: &mut Criterion) {
    // Approximate reference measurements are documented in src/gpu.rs. The GPU
    // names below describe distinct paths, rather than interchangeable timings:
    // - gpu_kernel: device-resident input, synchronized kernel only.
    // - gpu_end_to_end_prepared: prepared host bytes, device upload/allocation,
    //   kernel, and result readback.
    // - gpu_end_to_end_full: original Symbol/Spaceline representation through
    //   host preparation, then the prepared end-to-end path.
    let len = 100_000_000;
    let step = PowerOfTwo::from_usize(65536);
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let values = AVec::from_iter(ALIGN, (0..len).map(|_| rng.random_range(0u8..=1).into()));

    let mut group = criterion.benchmark_group("len_100m");

    group.bench_function("iterators", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                Spaceline::average_with_iterators(
                    SELECT_ONE,
                    black_box(&values_clone),
                    black_box(step),
                )
            },
        );
    });

    group.bench_function("simd32", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                Spaceline::average_with_simd::<32>(
                    SELECT_ONE,
                    black_box(&values_clone),
                    black_box(step),
                )
            },
        );
    });

    group.bench_function("simd64", |bencher| {
        bencher.iter_with_setup(
            || values.clone(),
            |values_clone| {
                Spaceline::average_with_simd::<64>(
                    SELECT_ONE,
                    black_box(&values_clone),
                    black_box(step),
                )
            },
        );
    });

    #[cfg(feature = "gpu")]
    {
        let expected = Spaceline::average_with_iterators(SELECT_ONE, &values, step);
        let mut gpu_average = GpuAverage::from_values(SELECT_ONE, &values, step).unwrap();
        gpu_average.run_kernel().unwrap();
        assert_eq!(
            gpu_average.readback().unwrap().as_slice(),
            expected.as_slice()
        );
        eprintln!(
            "gpu_breakdown {:?}",
            busy_beaver_blaze::GpuAverage::timing_breakdown(SELECT_ONE, &values, step).unwrap()
        );

        group.bench_function("gpu_kernel", |bencher| {
            bencher.iter(|| gpu_average.run_kernel().unwrap());
        });

        // Includes host padding/copy preparation, device allocation and H→D copy,
        // kernel execution, output readback, and synchronization.
        group.bench_function("gpu_end_to_end_full", |bencher| {
            bencher.iter(|| {
                average_with_gpu(SELECT_ONE, black_box(&values), black_box(step)).unwrap()
            });
        });

        let prepared_input = GpuAverage::prepare_input(&values, step);
        // Excludes host padding/copy preparation. The Arc clone is cheap; the
        // prepared bytes are still uploaded and device buffers are reallocated.
        group.bench_function("gpu_end_to_end_prepared", |bencher| {
            bencher.iter(|| {
                let mut gpu_average = GpuAverage::from_prepared_input(
                    SELECT_ONE,
                    prepared_input.clone(),
                    black_box(step),
                )
                .unwrap();
                gpu_average.run_kernel().unwrap();
                gpu_average.readback().unwrap()
            });
        });
    }

    group.finish();
}

fn resample(criterion: &mut Criterion) {
    let len = 4_000;
    let seed = 1;
    let mut rng = StdRng::seed_from_u64(seed);
    let pixels: AVec<Pixel, _> =
        AVec::from_iter(ALIGN, (0..len).map(|_| rng.random_range(0u8..=1).into()));

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
