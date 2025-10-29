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

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyBytes;

use crate::{PixelPolicy, PngDataIterator as RustPngDataIterator, BB5_CHAMP, BB6_CONTENDER};

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
            Ok([r * 17, g * 17, b * 17])  // 0xF -> 0xFF
        }
        _ => Err(format!("Invalid hex color format: {}. Expected #RRGGBB, RRGGBB, or #RGB", hex))
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
        )))
    }
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
    #[pyo3(signature = (early_stop, program, width, height, pixel_policy, frame_steps, colors=vec![], part_count=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        early_stop: u64,
        program: String,
        width: u32,
        height: u32,
        pixel_policy: String,
        frame_steps: Vec<u64>,
        colors: Vec<String>,
        part_count: usize,
    ) -> PyResult<Self> {
        // Parse pixel policy
        let pixel_policy = parse_pixel_policy(&pixel_policy)?;

        // Parse hex colors
        let colors_rgb: Result<Vec<[u8; 3]>, String> = colors
            .iter()
            .map(|s| parse_hex_color(s))
            .collect();
        let colors_rgb = colors_rgb.map_err(|e| PyValueError::new_err(e))?;

        // Use CPU count if part_count is 0
        let part_count = if part_count == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            part_count
        };

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

        Ok(Self {
            inner: Some(inner),
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<(u64, Py<PyBytes>)>> {
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

/// Python module for busy_beaver_blaze Rust bindings
#[pymodule]
fn _busy_beaver_blaze(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPngDataIterator>()?;
    
    // Export constants
    m.add("BB5_CHAMP", BB5_CHAMP)?;
    m.add("BB6_CONTENDER", BB6_CONTENDER)?;
    
    Ok(())
}
