//! PyO3 bindings for Python integration
//!
//! This module exposes the `PngDataIterator` to Python, following the nine rules
//! from "Nine Rules for Writing Python Extensions in Rust":
//! 1. Single repository with both Rust and Python
//! 2. Use maturin & PyO3 for translator functions
//! 3. Translator functions call "nice" Rust functions
//! 4. Memory preallocated in Python (not applicable here - iterator returns bytes)
//! 5. Translate Rust errors to Python exceptions
//! 6. Multithread with Rayon, release GIL
//! 7. Allow users to control thread count
//! 8. Dynamic types in Python → generic Rust functions
//! 9. Test both Rust and Python

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;

use crate::{
    BB5_CHAMP, BB6_CONTENDER, Machine, PixelPolicy, PngDataIterator as RustPngDataIterator,
    SpaceByTimeMachine as RustSpaceByTimeMachine,
};

/// Parse a hex color string to RGB tuple
///
/// Accepts formats: "#RRGGBB", "RRGGBB", "#RGB"
fn parse_hex_color(hex: &str) -> Result<[u8; 3], String> {
    let hex = hex.trim_start_matches('#');

    match hex.len() {
        6 => {
            // #RRGGBB format
            let r = u8::from_str_radix(&hex[0..2], 16)
                .map_err(|_| format!("Invalid red component in hex color: {}", hex))?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .map_err(|_| format!("Invalid green component in hex color: {}", hex))?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .map_err(|_| format!("Invalid blue component in hex color: {}", hex))?;
            Ok([r, g, b])
        }
        3 => {
            // #RGB format - expand to #RRGGBB
            let r = u8::from_str_radix(&hex[0..1], 16)
                .map_err(|_| format!("Invalid red component in hex color: {}", hex))?;
            let g = u8::from_str_radix(&hex[1..2], 16)
                .map_err(|_| format!("Invalid green component in hex color: {}", hex))?;
            let b = u8::from_str_radix(&hex[2..3], 16)
                .map_err(|_| format!("Invalid blue component in hex color: {}", hex))?;
            Ok([r * 17, g * 17, b * 17]) // 0xF -> 0xFF
        }
        _ => Err(format!(
            "Invalid hex color format: {}. Expected #RRGGBB, RRGGBB, or #RGB",
            hex
        )),
    }
}

/// Parse pixel policy string to enum
fn parse_pixel_policy(policy: &str) -> PyResult<PixelPolicy> {
    match policy.to_lowercase().as_str() {
        "binning" => Ok(PixelPolicy::Binning),
        "sampling" => Ok(PixelPolicy::Sampling),
        _ => Err(PyValueError::new_err(format!(
            "Invalid pixel_policy: '{}'. Expected 'binning' or 'sampling'",
            policy
        ))),
    }
}

/// Run a Turing machine program up to a step limit, returning steps executed and nonblank count.
#[pyfunction]
#[pyo3(signature = (program_text, step_limit))]
fn run_machine_steps(py: Python<'_>, program_text: &str, step_limit: u64) -> PyResult<(u64, u64)> {
    if step_limit == 0 {
        return Err(PyValueError::new_err("step_limit must be at least 1"));
    }

    let machine_result: Result<Machine, _> = program_text.parse();
    let machine = machine_result
        .map_err(|error| PyValueError::new_err(format!("Failed to parse program: {}", error)))?;

    let result = py.allow_threads(move || {
        let mut machine = machine;
        let mut steps_run = 0_u64;

        while steps_run < step_limit {
            match machine.next() {
                Some(_) => {
                    steps_run += 1;
                }
                None => break,
            }
        }

        let nonzero_count = u64::from(machine.count_nonblanks());
        (steps_run, nonzero_count)
    });
    Ok(result)
}

/// Iterator that generates PNG frames of Turing machine space-time diagrams.
///
/// This is a Python wrapper around the Rust `PngDataIterator`. It yields
/// `(step_index, png_bytes)` tuples, where `png_bytes` is the raw PNG image data.
///
/// # Parameters
///
/// * `early_stop` - Maximum step count to simulate
/// * `program` - Turing machine program string
/// * `width` - Target image width in pixels
/// * `height` - Target image height in pixels  
/// * `pixel_policy` - Either "binning" (average pixels) or "sampling" (pick pixels)
/// * `frame_steps` - List of step indices to generate frames for
/// * `colors` - List of hex color strings (e.g., ["#FF0000", "#00FF00"]), or empty for defaults
/// * `part_count` - Number of parallel work chunks (defaults to CPU count)
///
/// # Example
///
/// ```python
/// from busy_beaver_blaze import PngDataIterator, BB5_CHAMP
///
/// frame_steps = [0, 100, 1000, 10000, 100000]
/// iterator = PngDataIterator(
///     early_stop=1000000,
///     program=BB5_CHAMP,
///     width=1920,
///     height=1080,
///     pixel_policy="binning",
///     frame_steps=frame_steps,
///     colors=[],
///     part_count=8
/// )
///
/// for step_index, png_bytes in iterator:
///     print(f"Frame at step {step_index}, {len(png_bytes)} bytes")
/// ```
#[pyclass]
struct PyPngDataIterator {
    inner: Option<RustPngDataIterator>,
}

#[pymethods]
impl PyPngDataIterator {
    #[new]
    #[pyo3(signature = (frame_steps, resolution=(1920, 1080), early_stop=50_000_000, program=None, pixel_policy="binning", colors=vec![], part_count=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        frame_steps: Vec<u64>,
        resolution: (u32, u32),
        early_stop: u64,
        program: Option<String>,
        pixel_policy: &str,
        colors: Vec<String>,
        part_count: Option<usize>,
    ) -> PyResult<Self> {
        // Validate frame_steps is not empty
        if frame_steps.is_empty() {
            return Err(PyValueError::new_err(
                "frame_steps cannot be empty - specify at least one step index to capture",
            ));
        }

        let (width, height) = resolution;

        // Default program to BB6_CONTENDER
        let program = program.unwrap_or_else(|| BB6_CONTENDER.to_string());

        // Parse pixel policy
        let pixel_policy = parse_pixel_policy(pixel_policy)?;

        // Parse hex colors
        let colors_rgb: Result<Vec<[u8; 3]>, String> =
            colors.iter().map(|s| parse_hex_color(s)).collect();
        let colors_rgb = colors_rgb.map_err(|e| PyValueError::new_err(e))?;

        // Use CPU count if part_count is None
        let part_count = part_count.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });

        // Create iterator - release GIL during construction since it spawns threads
        let inner = py.allow_threads(|| {
            RustPngDataIterator::new(
                early_stop,
                part_count,
                &program,
                &colors_rgb,
                width,
                height,
                pixel_policy,
                &frame_steps,
            )
        });

        Ok(Self { inner: Some(inner) })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
    ) -> PyResult<Option<(u64, Py<PyBytes>)>> {
        let Some(ref mut inner) = slf.inner else {
            return Ok(None);
        };

        // Release GIL while waiting for next frame (iterator blocks on channel recv)
        let result = py.allow_threads(|| inner.next());

        match result {
            Some((step_index, png_data)) => {
                // Convert Vec<u8> to Python bytes
                let py_bytes = PyBytes::new_bound(py, &png_data).into();
                Ok(Some((step_index, py_bytes)))
            }
            None => Ok(None),
        }
    }
}

/// Interactive Turing machine with space-time visualization (mirrors WebAssembly API).
///
/// This class provides the same API as the JavaScript/WASM interface, allowing
/// notebooks to run indefinitely with periodic rendering. The machine can be
/// stopped at any time and continues to support recoloring.
///
/// # Parameters
///
/// * `program` - Turing machine program string
/// * `resolution` - Target (width, height) in pixels
/// * `binning` - True for pixel averaging, False for sampling
/// * `skip` - Number of initial steps to skip before visualization
///
/// # Example
///
/// ```python
/// from busy_beaver_blaze import SpaceByTimeMachine, BB5_CHAMP
///
/// # Create machine
/// machine = SpaceByTimeMachine(
///     program=BB5_CHAMP,
///     resolution=(1920, 1080),
///     binning=True,
///     skip=0
/// )
///
/// # Run for 0.1 seconds, max 1M steps
/// while machine.step_for_secs(0.1, early_stop=1_000_000):
///     png_bytes = machine.to_png()
///     # Display png_bytes...
///     
///     # Check if we want to stop
///     if machine.step_count() > 500_000:
///         break
///
/// # Get final state
/// print(f"Steps: {machine.step_count()}")
/// print(f"Nonblanks: {machine.count_nonblanks()}")
/// print(f"Halted: {machine.is_halted()}")
/// ```
#[pyclass]
struct PySpaceByTimeMachine {
    inner: RustSpaceByTimeMachine,
    /// Current color palette (15 bytes = 5 RGB colors)
    colors: Vec<u8>,
}

#[pymethods]
impl PySpaceByTimeMachine {
    #[new]
    #[pyo3(signature = (program, resolution=(800, 600), binning=true, skip=0, colors=vec![]))]
    fn new(
        program: String,
        resolution: (u32, u32),
        binning: bool,
        skip: u64,
        colors: Vec<String>,
    ) -> PyResult<Self> {
        let (width, height) = resolution;

        // Create machine
        let inner = RustSpaceByTimeMachine::from_str(&program, width, height, binning, skip)
            .map_err(|e| PyValueError::new_err(format!("Failed to create machine: {}", e)))?;

        // Parse colors or use empty (let Rust handle defaults)
        let colors_rgb: Vec<u8> = if colors.is_empty() {
            vec![]
        } else {
            // Parse hex colors
            let parsed: Result<Vec<[u8; 3]>, String> =
                colors.iter().map(|s| parse_hex_color(s)).collect();
            let parsed = parsed.map_err(|e| PyValueError::new_err(e))?;

            // Flatten to Vec<u8>
            parsed.into_iter().flatten().collect()
        };

        Ok(Self {
            inner,
            colors: colors_rgb,
        })
    }

    /// Step the machine for a specified duration.
    ///
    /// Returns True if more steps are available, False if halted or early_stop reached.
    ///
    /// # Arguments
    ///
    /// * `seconds` - Duration to run (e.g., 0.1)
    /// * `early_stop` - Optional maximum step count
    /// * `loops_per_time_check` - How often to check elapsed time (default: 10000)
    #[pyo3(signature = (seconds, early_stop=None, loops_per_time_check=10_000))]
    fn step_for_secs(
        &mut self,
        py: Python<'_>,
        seconds: f32,
        early_stop: Option<u64>,
        loops_per_time_check: u64,
    ) -> PyResult<bool> {
        // Release GIL while stepping
        let result = py.allow_threads(|| {
            self.inner
                .step_for_secs_js(seconds, early_stop, loops_per_time_check)
        });
        Ok(result)
    }

    /// Render current state to PNG bytes.
    ///
    /// Uses the color palette set during construction or via `set_colors()`.
    fn to_png(&mut self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        // Release GIL during rendering
        let png_data = py.allow_threads(|| self.inner.to_png(&self.colors));

        let png_data =
            png_data.map_err(|e| PyValueError::new_err(format!("PNG generation failed: {}", e)))?;

        Ok(PyBytes::new_bound(py, &png_data).into())
    }

    /// Update the color palette.
    ///
    /// # Arguments
    ///
    /// * `colors` - List of hex color strings (e.g., ["#FF0000", "#00FF00"])
    fn set_colors(&mut self, colors: Vec<String>) -> PyResult<()> {
        let parsed: Result<Vec<[u8; 3]>, String> =
            colors.iter().map(|s| parse_hex_color(s)).collect();
        let parsed = parsed.map_err(|e| PyValueError::new_err(e))?;

        self.colors = parsed.into_iter().flatten().collect();
        Ok(())
    }

    /// Get current step count (1-indexed).
    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    /// Count non-blank symbols on tape.
    fn count_nonblanks(&self) -> u32 {
        self.inner.count_nonblanks()
    }

    /// Check if machine has halted.
    fn is_halted(&self) -> bool {
        self.inner.is_halted()
    }

    /// Get the target resolution (width, height) for this machine.
    fn resolution(&self) -> (u32, u32) {
        let space_by_time = self.inner.space_time_layers.first();
        (space_by_time.x_goal, space_by_time.y_goal)
    }
}

/// Python module for busy_beaver_blaze Rust bindings
#[pymodule]
fn _busy_beaver_blaze(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_machine_steps, m)?)?;
    m.add_class::<PyPngDataIterator>()?;
    m.add_class::<PySpaceByTimeMachine>()?;

    // Export most commonly used program constants
    m.add("BB5_CHAMP", BB5_CHAMP)?;
    m.add("BB6_CONTENDER", BB6_CONTENDER)?;

    Ok(())
}
