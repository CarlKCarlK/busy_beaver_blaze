//! Experimental cuTile implementation of horizontal tape compression.
//!
//! The kernel computes one output pixel per `x_stride` symbols by counting the
//! selected symbol and scaling the count to `[0, 255]`. It intentionally does
//! not implement vertical/generation compression.
//!
//! The `len_100m` measurements below are approximate experimental results, not
//! performance guarantees. They were collected on the experiment machine with
//! 16 logical CPUs and an NVIDIA CUDA-capable GPU; the exact CPU/GPU models were
//! not recorded in the repository:
//!
//! ```text
//! CPU iterator             ~9.3 ms
//! CPU SIMD32               ~9.3 ms
//! CPU SIMD64               ~9.6 ms
//! GPU resident kernel      ~4.2 ms
//! GPU prepared end-to-end ~11.1 ms
//! GPU full end-to-end     ~67 ms
//! ```
//!
//! Portable SIMD is substantially faster on the smaller `large` workload but
//! provides little or no advantage on the 100-million-symbol workload, which
//! is consistent with the larger scan becoming memory-bandwidth dominated.
//! The resident horizontal GPU reduction is about 2.2x faster than the best CPU
//! implementation on that workload, but one-shot GPU offload is slower than
//! the CPU. The full path is dominated by host preparation because cuTile's
//! public host-copy API requires an owned `Arc<Vec<T>>`.
//!
//! cuTile does not currently expose a public operation to upload host data into
//! an already allocated device tensor, so device-allocation reuse cannot be
//! benchmarked without extending cuTile or using lower-level CUDA. Vertical
//! compression is adaptive and stateful: a future full GPU pipeline would need
//! to keep rows and their weights on-device and reproduce the pairwise,
//! power-of-two merge behavior in `Spacelines`.

use core::num::NonZeroU8;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aligned_vec::AVec;
use cutile::prelude::*;
use zerocopy::IntoBytes;

use crate::{Pixel, PowerOfTwo, Symbol, ALIGN};

#[cutile::module]
mod average_module {
    use cutile::core::*;

    #[cutile::entry]
    fn average<const STEP: i32>(
        output: &mut Tensor<u32, { [1] }>,
        input: &Tensor<u8, { [-1, -1] }>,
        selected_symbol: u8,
    ) {
        let program_index = get_tile_block_id().0;
        let input_part = input.partition(const_shape![1, STEP]);
        let input_tile: Tile<u8, { [1, STEP] }> = input_part.load([program_index, 0]);
        let select_tile: Tile<u8, { [1, STEP] }> = selected_symbol.broadcast(const_shape![1, STEP]);
        let selected: Tile<u32, { [1, STEP] }> = select(
            eq_tile(input_tile, select_tile),
            1u32.broadcast(const_shape![1, STEP]),
            0u32.broadcast(const_shape![1, STEP]),
        );
        let count: Tile<u32, { [1] }> = reduce_sum(selected, 1i32);
        let scaled = count * 255u32.broadcast(const_shape![1]);
        let average: Tile<u32, { [1] }> = scaled / (STEP as u32).broadcast(const_shape![1]);
        output.store(average);
    }
}

/// Device-resident input/output buffers for benchmarking the kernel alone.
pub struct GpuAverage {
    input: Tensor<u8>,
    output: Option<Tensor<u32>>,
    selected_symbol: u8,
    step_size: usize,
}

/// One-shot timings for diagnosing the end-to-end GPU path.
#[derive(Debug, Clone, Copy)]
pub struct GpuTimingBreakdown {
    pub host_preparation: Duration,
    pub input_allocation_and_host_to_device: Duration,
    pub output_allocation: Duration,
    pub kernel: Duration,
    pub device_to_host_and_synchronization: Duration,
    pub total: Duration,
}

fn prepared_input(values: &AVec<Symbol>, step: PowerOfTwo) -> (Arc<Vec<u8>>, usize, usize) {
    let step_size = step.as_usize();
    let output_len = step.div_ceil_into(values.len());
    let input_len = output_len * step_size;
    let mut input = Vec::with_capacity(input_len);
    input.extend_from_slice(values.as_slice().as_bytes());
    input.resize(input_len, 0);
    (Arc::new(input), output_len, step_size)
}

impl GpuAverage {
    /// Uploads and reshapes the input, and allocates the device output.
    pub fn from_values(
        select: NonZeroU8,
        values: &AVec<Symbol>,
        step: PowerOfTwo,
    ) -> Result<Self, String> {
        let (input, output_len, step_size) = prepared_input(values, step);
        assert!(output_len > 0, "GPU benchmark input must not be empty");

        let input = api::copy_host_vec_to_device(&input)
            .sync()
            .map_err(|error| error.to_string())?
            .reshape(&[output_len, step_size])
            .map_err(|error| error.to_string())?;
        let output = api::copy_host_vec_to_device(&Arc::new(vec![0u32; output_len]))
            .sync()
            .map_err(|error| error.to_string())?;

        Ok(Self {
            input,
            output: Some(output),
            selected_symbol: select.get(),
            step_size,
        })
    }

    /// Uploads already-prepared, padded byte input and allocates the output.
    pub fn from_prepared_input(
        select: NonZeroU8,
        input: Arc<Vec<u8>>,
        step: PowerOfTwo,
    ) -> Result<Self, String> {
        let step_size = step.as_usize();
        assert!(input.len() > 0, "GPU benchmark input must not be empty");
        assert!(
            step.divides_usize(input.len()),
            "Prepared GPU input must contain complete step groups"
        );
        let output_len = input.len() / step_size;

        let input = api::copy_host_vec_to_device(&input)
            .sync()
            .map_err(|error| error.to_string())?
            .reshape(&[output_len, step_size])
            .map_err(|error| error.to_string())?;
        let output = api::copy_host_vec_to_device(&Arc::new(vec![0u32; output_len]))
            .sync()
            .map_err(|error| error.to_string())?;

        Ok(Self {
            input,
            output: Some(output),
            selected_symbol: select.get(),
            step_size,
        })
    }

    /// Prepares the padded byte representation used by the GPU input tensor.
    pub fn prepare_input(values: &AVec<Symbol>, step: PowerOfTwo) -> Arc<Vec<u8>> {
        prepared_input(values, step).0
    }

    /// Launches and synchronizes the resident GPU kernel, excluding transfers.
    pub fn run_kernel(&mut self) -> Result<(), String> {
        let output = self.output.take().expect("GPU output must be present");
        let (output, _, _) =
            average_module::average(output.partition([1]), &self.input, self.selected_symbol)
                .generics(vec![self.step_size.to_string()])
                .sync()
                .map_err(|error| error.to_string())?;
        self.output = Some(output.unpartition());
        Ok(())
    }

    /// Measures one complete prepared-input operation after warming the JIT.
    ///
    /// cuTile's current host-copy helper allocates device storage and copies in one
    /// operation, so those two costs are intentionally reported together.
    pub fn timing_breakdown(
        select: NonZeroU8,
        values: &AVec<Symbol>,
        step: PowerOfTwo,
    ) -> Result<GpuTimingBreakdown, String> {
        // Compile and warm the specialization before starting the clock. The warm-up
        // is not part of any reported stage or of `total`.
        let mut warmup = Self::from_values(select, values, step)?;
        warmup.run_kernel()?;
        drop(warmup);

        let total_start = Instant::now();

        let host_start = Instant::now();
        let (input, output_len, step_size) = prepared_input(values, step);
        let host_preparation = host_start.elapsed();
        assert!(output_len > 0, "GPU benchmark input must not be empty");

        let input_start = Instant::now();
        let input = api::copy_host_vec_to_device(&input)
            .sync()
            .map_err(|error| error.to_string())?
            .reshape(&[output_len, step_size])
            .map_err(|error| error.to_string())?;
        let input_allocation_and_host_to_device = input_start.elapsed();

        let output_start = Instant::now();
        let output = api::copy_host_vec_to_device(&Arc::new(vec![0u32; output_len]))
            .sync()
            .map_err(|error| error.to_string())?;
        let output_allocation = output_start.elapsed();

        let mut gpu_average = Self {
            input,
            output: Some(output),
            selected_symbol: select.get(),
            step_size,
        };

        let kernel_start = Instant::now();
        gpu_average.run_kernel()?;
        let kernel = kernel_start.elapsed();

        let readback_start = Instant::now();
        gpu_average.readback()?;
        let device_to_host_and_synchronization = readback_start.elapsed();

        Ok(GpuTimingBreakdown {
            host_preparation,
            input_allocation_and_host_to_device,
            output_allocation,
            kernel,
            device_to_host_and_synchronization,
            total: total_start.elapsed(),
        })
    }

    /// Synchronizes a device-to-host copy of the current output.
    pub fn readback(&self) -> Result<AVec<Pixel>, String> {
        let host_output = self
            .output
            .as_ref()
            .expect("GPU output must be present")
            .dup()
            .sync()
            .map_err(|error| error.to_string())?
            .to_host_vec()
            .sync()
            .map_err(|error| error.to_string())?;
        Ok(AVec::from_iter(
            ALIGN,
            host_output.into_iter().map(|value| {
                let value = u8::try_from(value).expect("GPU pixel value must fit in u8");
                Pixel::from(value)
            }),
        ))
    }
}

/// Computes the same selected-symbol averages as `Spaceline::average_with_simd`.
///
/// This is an experimental, synchronous host-to-device/device-to-host path. The
/// input is padded with zero symbols so a final partial group is divided by the
/// complete `step`, matching the CPU implementations.
#[must_use]
pub fn average_with_gpu(
    select: NonZeroU8,
    values: &AVec<Symbol>,
    step: PowerOfTwo,
) -> Result<AVec<Pixel>, String> {
    let output_len = step.div_ceil_into(values.len());
    if output_len == 0 {
        return Ok(AVec::new(ALIGN));
    }

    let mut gpu_average = GpuAverage::from_values(select, values, step)?;
    gpu_average.run_kernel()?;
    gpu_average.readback()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Spaceline;

    fn check(values: &[u8], step: usize) {
        let values = AVec::from_iter(ALIGN, values.iter().copied().map(Symbol::from));
        let step = PowerOfTwo::from_usize(step);
        let expected = Spaceline::average_with_iterators(NonZeroU8::new(1).unwrap(), &values, step);
        let actual = average_with_gpu(NonZeroU8::new(1).unwrap(), &values, step).unwrap();
        assert_eq!(actual.as_slice(), expected.as_slice());
    }

    #[test]
    fn prepared_input_is_byte_equivalent_and_padded() {
        let values = AVec::from_iter(ALIGN, [0u8, 1, 2, 1].into_iter().map(Symbol::from));
        let input = GpuAverage::prepare_input(&values, PowerOfTwo::from_usize(4));
        assert_eq!(input.as_slice(), &[0, 1, 2, 1]);

        let input = GpuAverage::prepare_input(&values, PowerOfTwo::from_usize(8));
        assert_eq!(input.as_slice(), &[0, 1, 2, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn matches_cpu_for_edge_and_mixed_inputs() {
        for step in [1, 2, 4, 8, 16] {
            check(&[], step);
            check(&[0], step);
            check(&[1], step);
            check(&[0, 1, 2, 1, 3, 0, 1, 2, 1], step);
            check(&vec![0; step], step);
            check(&vec![1; step], step);
            check(
                &(0..step * 3 + 1)
                    .map(|value| (value % 4) as u8)
                    .collect::<Vec<_>>(),
                step,
            );
        }
    }
}
