//! Experimental cuda-oxide SIMT implementation of horizontal tape compression.
//!
//! Build and run this implementation through `cargo oxide`; the ordinary
//! Cargo backend cannot compile the `#[cuda_module]` device artifact.

use core::num::NonZeroU8;

use aligned_vec::AVec;
use cuda_core::simt::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig, PinnedHostBuffer};
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;
use zerocopy::IntoBytes;

use crate::{Pixel, PowerOfTwo, Symbol, ALIGN};

#[cuda_module]
mod horizontal_module {
    use super::*;

    #[kernel]
    pub fn horizontal_average(
        input: &[u8],
        selected_symbol: u8,
        step_size: usize,
        mut output: DisjointSlice<u8>,
    ) {
        let pixel_index = thread::index_1d();
        let pixel_index_raw = pixel_index.get();
        if pixel_index_raw >= output.len() {
            return;
        }

        let first_symbol = pixel_index_raw * step_size;
        let mut selected_count = 0u32;
        let mut symbol_index = 0usize;
        while symbol_index < step_size {
            if input[first_symbol + symbol_index] == selected_symbol {
                selected_count += 1;
            }
            symbol_index += 1;
        }
        let pixel = (selected_count * 255 / step_size as u32) as u8;
        if let Some(output_pixel) = output.get_mut(pixel_index) {
            *output_pixel = pixel;
        }
    }
}

/// Persistent cuda-oxide context, module, and device buffers.
pub struct CudaOxideAverage {
    context: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<CudaStream>,
    module: horizontal_module::LoadedModule,
    input: DeviceBuffer<u8>,
    output: DeviceBuffer<u8>,
    host_output: Vec<u8>,
    real_input_len: usize,
    output_len: usize,
    step_size: usize,
    selected_symbol: u8,
}

impl CudaOxideAverage {
    /// Allocates all CUDA state once. The input buffer includes only the small
    /// final-group tail padding; timed refills copy the original tape bytes.
    pub fn new(select: NonZeroU8, values: &AVec<Symbol>, step: PowerOfTwo) -> Result<Self, String> {
        let real_input_len = values.len();
        let step_size = step.as_usize();
        let output_len = step.div_ceil_into(real_input_len);
        if output_len == 0 {
            return Err("cuda-oxide benchmark input must not be empty".to_owned());
        }
        let input_len = output_len * step_size;
        let context = CudaContext::new(0).map_err(|error| error.to_string())?;
        let stream = context.default_stream();
        let module = horizontal_module::load(&context).map_err(|error| error.to_string())?;
        let input = DeviceBuffer::zeroed(&stream, input_len).map_err(|error| error.to_string())?;
        let output =
            DeviceBuffer::zeroed(&stream, output_len).map_err(|error| error.to_string())?;
        let host_output = vec![0u8; output_len];

        let mut average = Self {
            context,
            stream,
            module,
            input,
            output,
            host_output,
            real_input_len,
            output_len,
            step_size,
            selected_symbol: select.get(),
        };
        average.refill(values)?;
        Ok(average)
    }

    /// Refills the existing device input allocation, including only the
    /// zero-filled tail for a non-multiple input.
    pub fn refill(&mut self, values: &AVec<Symbol>) -> Result<(), String> {
        assert_eq!(values.len(), self.real_input_len);
        let values = values.as_slice().as_bytes();
        if !values.is_empty() {
            let copy_result = unsafe {
                cuda_core::simt::memory::memcpy_htod_async(
                    self.input.cu_deviceptr(),
                    values.as_ptr(),
                    values.len(),
                    self.stream.cu_stream(),
                )
            };
            copy_result.map_err(|error| error.to_string())?;
        }
        let padding_len = self.output_len * self.step_size - values.len();
        if padding_len != 0 {
            let tail_ptr = self.input.cu_deviceptr() + values.len() as u64;
            unsafe {
                cuda_core::simt::memory::memset_d8_async(
                    tail_ptr,
                    0,
                    padding_len,
                    self.stream.cu_stream(),
                )
            }
            .map_err(|error| error.to_string())?;
        }
        self.stream.synchronize().map_err(|error| error.to_string())
    }

    /// Launches the resident-input/resident-output kernel and synchronizes.
    pub fn run_kernel(&mut self) -> Result<(), String> {
        // SAFETY: one-dimensional launch has one thread per output pixel and
        // both buffers cover every access made by the kernel.
        unsafe {
            self.module.horizontal_average(
                &self.stream,
                LaunchConfig::for_num_elems(self.output_len as u32),
                &self.input,
                self.selected_symbol,
                self.step_size,
                &mut self.output,
            )
        }
        .map_err(|error| error.to_string())?;
        self.stream.synchronize().map_err(|error| error.to_string())
    }

    /// Copies the small result into the allocation-free host result buffer.
    pub fn readback(&mut self) -> Result<(), String> {
        self.output
            .copy_to_host(&self.stream, &mut self.host_output)
            .map_err(|error| error.to_string())
    }

    /// Runs the reusable-allocation path: H→D refill, kernel, and D→H result.
    pub fn refill_run_readback(&mut self, values: &AVec<Symbol>) -> Result<(), String> {
        self.refill(values)?;
        self.run_kernel()?;
        self.readback()
    }

    /// Returns the most recent result in the crate's pixel representation.
    pub fn pixels(&self) -> AVec<Pixel> {
        AVec::from_iter(ALIGN, self.host_output.iter().copied().map(Pixel::from))
    }

    /// Allocates one reusable pinned staging buffer and measures the same path.
    pub fn refill_run_readback_pinned(
        &mut self,
        values: &AVec<Symbol>,
        pinned_input: &mut PinnedHostBuffer<u8>,
    ) -> Result<(), String> {
        let values = values.as_slice().as_bytes();
        pinned_input.as_mut_slice()[..values.len()].copy_from_slice(values);
        let padding_len = self.output_len * self.step_size - values.len();
        pinned_input.as_mut_slice()[values.len()..values.len() + padding_len].fill(0);
        unsafe {
            self.input
                .copy_from_pinned_host_async(&self.stream, pinned_input)
        }
        .map_err(|error| error.to_string())?;
        self.run_kernel()?;
        self.readback()
    }

    pub fn context(&self) -> &std::sync::Arc<CudaContext> {
        &self.context
    }

    pub fn input_len(&self) -> usize {
        self.output_len * self.step_size
    }
}

/// One-shot cuda-oxide operation, including context/module/buffer setup.
pub fn average_with_cuda_oxide(
    select: NonZeroU8,
    values: &AVec<Symbol>,
    step: PowerOfTwo,
) -> Result<AVec<Pixel>, String> {
    if values.is_empty() {
        return Ok(AVec::new(ALIGN));
    }
    let mut average = CudaOxideAverage::new(select, values, step)?;
    average.run_kernel()?;
    average.readback()?;
    Ok(average.pixels())
}
