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

macro_rules! benchmark_average_variants {
    ($group:expr, $benchmark_prefix:expr, $values:expr, $step:expr) => {{
        let expected = Spaceline::average_with_iterators(SELECT_ONE, &$values, $step);
        assert_eq!(
            Spaceline::average_with_simd::<32>(SELECT_ONE, &$values, $step).as_slice(),
            expected.as_slice()
        );
        assert_eq!(
            Spaceline::average_with_simd::<64>(SELECT_ONE, &$values, $step).as_slice(),
            expected.as_slice()
        );

        $group.bench_function(format!("{}/iterators", $benchmark_prefix), |bencher| {
            bencher.iter_with_setup(
                || $values.clone(),
                |values_clone| {
                    Spaceline::average_with_iterators(
                        SELECT_ONE,
                        black_box(&values_clone),
                        black_box($step),
                    )
                },
            );
        });

        $group.bench_function(format!("{}/simd32", $benchmark_prefix), |bencher| {
            bencher.iter_with_setup(
                || $values.clone(),
                |values_clone| {
                    Spaceline::average_with_simd::<32>(
                        SELECT_ONE,
                        black_box(&values_clone),
                        black_box($step),
                    )
                },
            );
        });

        $group.bench_function(format!("{}/simd64", $benchmark_prefix), |bencher| {
            bencher.iter_with_setup(
                || $values.clone(),
                |values_clone| {
                    Spaceline::average_with_simd::<64>(
                        SELECT_ONE,
                        black_box(&values_clone),
                        black_box($step),
                    )
                },
            );
        });
    }};
}

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

fn input_size_sweep(criterion: &mut Criterion) {
    const STEP: usize = 512;
    const SIZES: &[usize] = &[
        64 * 1024,
        256 * 1024,
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        100_000_000,
    ];

    let maximum_len = *SIZES.last().unwrap();
    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let all_values: AVec<_, aligned_vec::RuntimeAlign> = AVec::from_iter(
        ALIGN,
        (0..maximum_len).map(|_| rng.random_range(0u8..=1).into()),
    );
    let mut group = criterion.benchmark_group("input_size_sweep_step_512");

    for &len in SIZES {
        let values = AVec::from_iter(ALIGN, all_values[..len].iter().copied());
        let benchmark_prefix = format_size(len);
        benchmark_average_variants!(
            group,
            benchmark_prefix,
            values,
            PowerOfTwo::from_usize_unchecked(STEP)
        );
    }

    group.finish();
}

fn step_sweep(criterion: &mut Criterion) {
    const LEN: usize = 100_000_000;
    const STEPS: &[usize] = &[512, 1_024, 4_096, 16_384, 65_536];

    let seed = 0;
    let mut rng = StdRng::seed_from_u64(seed);
    let values = AVec::from_iter(
        ALIGN,
        (0..LEN).map(|_| rng.random_range(0u8..=1).into()),
    );
    let mut group = criterion.benchmark_group("step_sweep_len_100m");

    for &step in STEPS {
        benchmark_average_variants!(
            group,
            step.to_string(),
            values,
            PowerOfTwo::from_usize_unchecked(step)
        );
    }

    group.finish();
}

fn format_size(len: usize) -> String {
    if len == 64 * 1024 {
        "64KiB".to_owned()
    } else if len == 256 * 1024 {
        "256KiB".to_owned()
    } else if len == 1024 * 1024 {
        "1MiB".to_owned()
    } else if len == 4 * 1024 * 1024 {
        "4MiB".to_owned()
    } else if len == 16 * 1024 * 1024 {
        "16MiB".to_owned()
    } else if len == 64 * 1024 * 1024 {
        "64MiB".to_owned()
    } else if len == 100_000_000 {
        "100MB".to_owned()
    } else {
        unreachable!("input-size sweep contains an unformatted size")
    }
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

criterion_group!(
    benches,
    small,
    large,
    len_100m,
    input_size_sweep,
    step_sweep,
    resample
);
criterion_main!(benches);
