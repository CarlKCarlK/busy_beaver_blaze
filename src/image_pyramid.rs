//! Experimental cumulative native-layer checkpoints and palette-independent reconstruction.
use aligned_vec::AVec;
use instant::Instant;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::space_scale::{SpaceScale, SpatialMap, SpatialRow, quantize};
use crate::space_time_layers::SpaceTimeLayers;
use crate::{ALIGN, Pixel, PixelPolicy, SpaceByTime, SpaceByTimeMachine, Spaceline};

/// Time mappings available when history capture is enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeScale {
    /// Reproduce the reference renderer's bins and blank padding.
    Linear,
    /// Map normalized image position to squared simulation time.
    SquareRoot,
    /// Continuous logarithmic time, with a one-step origin scale.
    Logarithmic,
}

impl TimeScale {
    fn parse(scale: &str) -> Result<Self, String> {
        match scale {
            "linear-b" => Ok(Self::Linear),
            "sqrt" => Ok(Self::SquareRoot),
            "log" => Ok(Self::Logarithmic),
            _ => Err(format!("Unknown pyramid time scale: {scale}")),
        }
    }

    fn boundary(self, position: f64, total: f64, origin: f64) -> f64 {
        assert!(origin > 0.0 && origin.is_finite());
        if position == 0.0 {
            return 0.0;
        }
        if position == 1.0 {
            return total;
        }
        match self {
            Self::Linear => total * position,
            Self::SquareRoot => total * position * position,
            Self::Logarithmic => origin * (position * (total / origin).ln_1p()).exp_m1(),
        }
    }
}

struct PyramidLevel {
    // Number of retained time samples, including the initial tape at time zero.
    sample_count: u64,
    layers: SpaceTimeLayers,
}

/// Cumulative checkpoints at powers of two simulated transitions; no machine/tape copies.
#[derive(Default)]
pub struct ImagePyramid {
    levels: Vec<PyramidLevel>,
}

impl ImagePyramid {
    fn capture(&mut self, active: &SpaceByTimeMachine) {
        let step_index = active.step_index();
        if step_index == 0 || step_index.is_power_of_two() {
            assert!(
                self.levels
                    .last()
                    .is_none_or(|level| level.sample_count < active.step_count())
            );
            self.levels.push(PyramidLevel {
                sample_count: active.step_count(),
                layers: active.space_time_layers.clone(),
            });
        }
    }

    fn source<'a>(&'a self, active: &'a SpaceTimeLayers, end: f64) -> &'a SpaceTimeLayers {
        self.levels
            .iter()
            .find(|level| level.sample_count as f64 >= end)
            .map_or(active, |level| &level.layers)
    }

    #[cfg(test)]
    fn reconstruct(&self, active: &SpaceByTimeMachine, scale: TimeScale) -> SpaceTimeLayers {
        self.reconstruct_with_space(active, scale, SpaceScale::Linear)
    }

    fn reconstruct_with_space(
        &self,
        active: &SpaceByTimeMachine,
        scale: TimeScale,
        space_scale: SpaceScale,
    ) -> SpaceTimeLayers {
        let first = active.space_time_layers.first();
        let tape = active.machine.tape();
        let x_stride = Spaceline::find_x_stride(
            tape.negative.len(),
            tape.nonnegative.len(),
            first.x_goal as usize,
        );
        let negative_len = x_stride.div_ceil_into(tape.negative.len());
        let nonnegative_len = x_stride.div_ceil_into(tape.nonnegative.len());
        let width = negative_len + nonnegative_len;
        let spatial_map = (space_scale != SpaceScale::Linear)
            .then(|| SpatialMap::new(space_scale, negative_len, nonnegative_len, x_stride));
        let mut spatial_row = spatial_map.as_ref().map(|_| SpatialRow::default());
        let internal_height = first.spacelines.len();
        let height = if scale == TimeScale::Linear || internal_height < 2 * first.y_goal as usize {
            internal_height
        } else {
            first.y_goal as usize
        };
        let total = active.step_count() as f64;
        let mut output = SpaceTimeLayers::default();
        for (select, layer) in &active.space_time_layers {
            let mut reconstructed =
                SpaceByTime::new0(*select, layer.x_goal, layer.y_goal, PixelPolicy::Binning);
            reconstructed.spacelines.main.clear();
            for row_index in 0..height {
                let (start, end) = if scale == TimeScale::Linear {
                    // u128 avoids overflowing the padded endpoint near u64::MAX.
                    let stride = u128::from(layer.y_stride.as_u64());
                    (
                        (row_index as u128 * stride) as f64,
                        ((row_index as u128 + 1) * stride) as f64,
                    )
                } else {
                    (
                        scale.boundary(row_index as f64 / height as f64, total, 1.0),
                        scale.boundary((row_index + 1) as f64 / height as f64, total, 1.0),
                    )
                };
                assert!(end > start && start.is_finite() && end.is_finite());
                let source = &self.source(&active.space_time_layers, end.min(total))[*select];
                let mut sums = vec![0.0f64; width];
                // Buffer entries retain their true weights: do not pad them before weighting.
                for (line, weight) in source
                    .spacelines
                    .main
                    .iter()
                    .map(|line| (line, source.y_stride))
                    .chain(
                        source
                            .spacelines
                            .buffer0
                            .iter()
                            .map(|(line, weight)| (line, *weight)),
                    )
                {
                    let line_start = (line.time - source.skip) as f64;
                    let line_end =
                        (u128::from(line.time - source.skip) + u128::from(weight.as_u64())) as f64;
                    let overlap = end.min(line_end).min(total) - start.max(line_start);
                    if overlap <= 0.0 {
                        continue;
                    }
                    let mut line = line.clone();
                    assert!(line.x_stride <= x_stride);
                    #[cfg(feature = "simd")]
                    line.compress_x_if_needed_simd(x_stride);
                    #[cfg(not(feature = "simd"))]
                    line.compress_x_if_needed_no_simd(x_stride);
                    if let Some(spatial_map) = &spatial_map {
                        let spatial_row = spatial_row
                            .as_mut()
                            .expect("spatial map must have a row workspace");
                        spatial_row.replace_line(&line);
                        spatial_map.accumulate(spatial_row, overlap, &mut sums);
                        continue;
                    }
                    for (pixel_index, pixel) in line.negative.iter().enumerate() {
                        sums[negative_len - pixel_index - 1] += f64::from(pixel.as_u8()) * overlap;
                    }
                    for (pixel_index, pixel) in line.nonnegative.iter().enumerate() {
                        sums[negative_len + pixel_index] += f64::from(pixel.as_u8()) * overlap;
                    }
                }
                let pixels: Vec<Pixel> = sums
                    .into_iter()
                    .map(|sum| {
                        Pixel::from(if spatial_map.is_some() {
                            quantize(sum / (end - start))
                        } else {
                            (sum / (end - start)).floor().clamp(0.0, 255.0) as u8
                        })
                    })
                    .collect();
                let mut negative = AVec::with_capacity(ALIGN, negative_len);
                for pixel in pixels[..negative_len].iter().rev() {
                    negative.push(*pixel);
                }
                let mut nonnegative = AVec::with_capacity(ALIGN, nonnegative_len);
                nonnegative.extend_from_slice(&pixels[negative_len..]);
                reconstructed.spacelines.main.push(Spaceline {
                    select: *select,
                    x_stride,
                    negative,
                    nonnegative,
                    time: row_index as u64,
                    pixel_policy: PixelPolicy::Binning,
                });
            }
            reconstructed.vis_step = height as u64 - 1;
            output.insert(*select, reconstructed);
        }
        output
    }
}

/// Separate experimental stepping path: the reference machine has no history checks.
#[wasm_bindgen]
pub struct PyramidMachine {
    active: SpaceByTimeMachine,
    pyramid: ImagePyramid,
    scale: TimeScale,
    space_scale: SpaceScale,
}

#[wasm_bindgen]
impl PyramidMachine {
    /// Start an experimental Quality run with native multilayer history capture.
    #[wasm_bindgen(constructor)]
    pub fn new(program: &str, x_goal: u32, y_goal: u32, scale: &str) -> Result<Self, String> {
        if x_goal < 2 || y_goal == 0 {
            return Err("Image goals must be width >= 2 and height >= 1".into());
        }
        let mut machine = Self {
            active: SpaceByTimeMachine::from_str(program, x_goal, y_goal, true, 0)?,
            pyramid: ImagePyramid::default(),
            scale: TimeScale::parse(scale)?,
            space_scale: SpaceScale::Linear,
        };
        machine.pyramid.capture(&machine.active);
        Ok(machine)
    }

    /// Change reconstruction without rerunning the simulation.
    pub fn set_time_scale(&mut self, scale: &str) -> Result<(), String> {
        self.scale = TimeScale::parse(scale)?;
        Ok(())
    }

    /// Change the horizontal coordinate map without stepping or capturing history.
    pub fn set_space_scale(&mut self, scale: &str) -> Result<(), String> {
        self.space_scale = SpaceScale::parse(scale)?;
        Ok(())
    }

    /// Step in bounded batches, capturing exact checkpoints even inside a batch.
    pub fn step_for_secs(
        &mut self,
        seconds: f32,
        early_stop: Option<u64>,
        loops_per_time_check: u64,
    ) -> bool {
        assert!(loops_per_time_check > 0);
        let start = Instant::now();
        loop {
            for _step_index in 0..loops_per_time_check {
                if early_stop.is_some_and(|limit| self.step_count() >= limit) || !self.step() {
                    return false;
                }
            }
            if start.elapsed().as_secs_f32() >= seconds {
                return true;
            }
        }
    }

    /// Advance one transition using the same halt convention as Linear A.
    pub fn step(&mut self) -> bool {
        if self.active.is_halted() || self.active.next().is_none() {
            return false;
        }
        self.pyramid.capture(&self.active);
        true
    }

    /// Encode reconstructed symbol layers using the existing PNG encoder.
    pub fn to_png(&self, colors: &[u8]) -> Result<Vec<u8>, String> {
        let (colors, remainder) = colors.as_chunks::<3>();
        if !remainder.is_empty() || colors.len() < 2 {
            return Err("Provide at least two RGB triplets".into());
        }
        let mut layers =
            self.pyramid
                .reconstruct_with_space(&self.active, self.scale, self.space_scale);
        let first = self.active.space_time_layers.first();
        layers
            .png_data_and_packed_data(
                colors,
                self.active.machine.tape.negative.len(),
                self.active.machine.tape.nonnegative.len(),
                (first.x_goal as usize, first.y_goal as usize),
            )
            .map(|(png, _, _, _)| png)
            .map_err(|error| error.to_string())
    }

    /// Render the active reference image while keeping pyramid history available.
    pub fn linear_png(&mut self, colors: &[u8]) -> Result<Vec<u8>, String> {
        match self.space_scale {
            SpaceScale::Linear => self.active.to_png(colors),
            SpaceScale::SquareRoot => self.active.to_spatial_png(colors, "sqrt"),
            SpaceScale::Logarithmic => self.active.to_spatial_png(colors, "log"),
            SpaceScale::Square => self.active.to_spatial_png(colors, "square"),
            SpaceScale::Exponential => self.active.to_spatial_png(colors, "exp"),
        }
    }

    /// Current sample count, matching the reference API (initial row included).
    pub fn step_count(&self) -> u64 {
        self.active.step_count()
    }
    /// Number of nonblank tape cells.
    pub fn count_nonblanks(&self) -> u32 {
        self.active.count_nonblanks()
    }
    /// Whether the underlying program halted.
    pub fn is_halted(&self) -> bool {
        self.active.is_halted()
    }
    /// Number of cumulative checkpoints, including the initial blank image.
    pub fn checkpoint_count(&self) -> u32 {
        self.pyramid.levels.len() as u32
    }
    /// Estimated retained layer bytes, including allocation capacity and row structures.
    /// Excludes allocator metadata, tape, PNG buffers and transient reconstruction storage.
    pub fn retained_bytes(&self) -> f64 {
        let layer_bytes = |layers: &SpaceTimeLayers| {
            layers
                .values()
                .map(SpaceByTime::retained_bytes)
                .sum::<usize>()
        };
        (layer_bytes(&self.active.space_time_layers)
            + self
                .pyramid
                .levels
                .iter()
                .map(|level| layer_bytes(&level.layers))
                .sum::<usize>()) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BB2_CHAMP, BB4_CHAMP, BB5_CHAMP, BB6_CONTENDER};

    // The web state-color transform assigns a different nonzero symbol to each writing state.
    const BB5_WRITING_STATE: &str = "1RB1LC1LC1LC1LC1LC_2RC2RB2RB2RB2RB2RB_3RD0LE0LE0LE0LE0LE_4LA4LD4LD4LD4LD4LD_5RH0LA0LA0LA0LA0LA";

    fn packed(machine: &PyramidMachine, scale: TimeScale) -> (u32, u32, Vec<AVec<u8>>) {
        let mut layers = machine.pyramid.reconstruct(&machine.active, scale);
        let first = machine.active.space_time_layers.first();
        let (_, width, height, pixels) = layers
            .png_data_and_packed_data(
                &[[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]],
                machine.active.machine.tape.negative.len(),
                machine.active.machine.tape.nonnegative.len(),
                (first.x_goal as usize, first.y_goal as usize),
            )
            .unwrap();
        (width, height, pixels)
    }

    #[test]
    fn linear_comparison_report() {
        let mut total_difference = 0u64;
        let mut total_pixels = 0u64;
        let mut differing = 0u64;
        let mut maximum = 0u8;
        for program in [
            BB2_CHAMP,
            BB4_CHAMP,
            BB5_CHAMP,
            BB6_CONTENDER,
            "1RB2LA3RA0LB_3LA2RB1LB0RA",
        ] {
            for (width, height) in [(2, 1), (5, 3), (16, 8), (64, 32)] {
                let mut machine = PyramidMachine::new(program, width, height, "linear-b").unwrap();
                let mut baseline =
                    SpaceByTimeMachine::from_str(program, width, height, true, 0).unwrap();
                for step_index in 0u64..4097 {
                    if step_index < 40 || step_index % 127 == 0 || step_index.is_power_of_two() {
                        let (_, reference_width, reference_height, reference) = baseline
                            .png_data_and_packed_data(&[
                                [255, 255, 255],
                                [255, 0, 0],
                                [0, 255, 0],
                                [0, 0, 255],
                            ]);
                        let (actual_width, actual_height, actual) =
                            packed(&machine, TimeScale::Linear);
                        assert_eq!(
                            (actual_width, actual_height),
                            (reference_width, reference_height)
                        );
                        for (reference, actual) in reference
                            .iter()
                            .flat_map(|layer| layer.iter())
                            .zip(actual.iter().flat_map(|layer| layer.iter()))
                        {
                            let difference = reference.abs_diff(*actual);
                            maximum = maximum.max(difference);
                            total_difference += u64::from(difference);
                            differing += u64::from(difference != 0);
                            total_pixels += 1;
                        }
                    }
                    if !machine.step() {
                        break;
                    }
                    assert!(baseline.next().is_some());
                }
            }
        }
        let mean = total_difference as f64 / total_pixels as f64;
        println!(
            "Linear B raw pixels: max={maximum}, mean={mean:.6}, differing={differing}/{total_pixels} ({:.3}%)",
            100.0 * differing as f64 / total_pixels as f64
        );
        assert!(maximum <= 2, "maximum={maximum}");
        assert!(mean < 0.05, "mean={mean}");
    }

    #[test]
    fn million_step_comparison_report() {
        for (name, program) in [
            ("BB5", BB5_CHAMP),
            ("BB6", BB6_CONTENDER),
            ("BB5 writing state", BB5_WRITING_STATE),
            ("multisymbol", "1RB2LA3RA0LB_3LA2RB1LB0RA"),
        ] {
            let mut machine = PyramidMachine::new(program, 96, 63, "linear-b").unwrap();
            let mut baseline = SpaceByTimeMachine::from_str(program, 96, 63, true, 0).unwrap();
            let mut sum = 0u64;
            let mut count = 0u64;
            let mut differing = 0u64;
            let mut maximum = 0u8;
            for step_index in 0u64..=1_000_001 {
                if step_index < 3
                    || step_index % 65537 == 0
                    || step_index.is_power_of_two()
                    || (step_index > 0 && (step_index - 1).is_power_of_two())
                {
                    let (_, width, height, reference) = baseline.png_data_and_packed_data(&[
                        [255, 255, 255],
                        [255, 0, 0],
                        [0, 255, 0],
                        [0, 0, 255],
                    ]);
                    let (actual_width, actual_height, actual) = packed(&machine, TimeScale::Linear);
                    assert_eq!((width, height), (actual_width, actual_height));
                    for (reference, actual) in reference
                        .iter()
                        .flat_map(|layer| layer.iter())
                        .zip(actual.iter().flat_map(|layer| layer.iter()))
                    {
                        let difference = reference.abs_diff(*actual);
                        sum += u64::from(difference);
                        count += 1;
                        differing += u64::from(difference != 0);
                        maximum = maximum.max(difference);
                    }
                }
                if !machine.step() {
                    break;
                }
                assert!(baseline.next().is_some());
            }
            let mean = sum as f64 / count as f64;
            println!(
                "{name}: max={maximum} mean={mean:.6} differing={differing}/{count} ({:.4}%)",
                100.0 * differing as f64 / count as f64
            );
            assert!(maximum <= 4, "{name}: max={maximum}");
            assert!(mean < 0.03, "{name}: mean={mean}");
        }
    }

    #[test]
    fn exact_checkpoints_and_immutable_rendering() {
        let mut machine = PyramidMachine::new("1RA1RA", 4, 3, "sqrt").unwrap();
        for step_index in 0u64..130 {
            assert_eq!(machine.active.step_index(), step_index);
            assert_eq!(
                machine.checkpoint_count(),
                if step_index == 0 {
                    1
                } else {
                    2 + step_index.ilog2()
                }
            );
            for level in &machine.pyramid.levels {
                assert_eq!(level.layers.step_index() + 1, level.sample_count);
            }
            let before = machine.retained_bytes();
            let png = machine.to_png(&[255, 255, 255, 255, 0, 0]).unwrap();
            assert_eq!(png, machine.to_png(&[255, 255, 255, 255, 0, 0]).unwrap());
            assert_eq!(before, machine.retained_bytes());
            machine.step();
        }
    }

    #[test]
    fn independent_small_time_averages() {
        // March right leaving ones. Four samples: blank, 100, 110, 111.
        let mut machine = PyramidMachine::new("1RA1RA", 16, 4, "sqrt").unwrap();
        for _step_index in 0..3 {
            assert!(machine.step());
        }
        let (width, height, layers) = packed(&machine, TimeScale::SquareRoot);
        assert_eq!((width, height), (3, 4));
        // Square boundaries [0,.25,1,2.25,4]: last two rows average known tape states.
        assert_eq!(
            layers[0].as_slice(),
            &[0, 0, 0, 0, 0, 0, 255, 51, 0, 255, 255, 145]
        );
        let (width, height, layers) = packed(&machine, TimeScale::Logarithmic);
        let boundaries: Vec<f64> = (0..=4)
            .map(|row| 5f64.powf(row as f64 / 4.0) - 1.0)
            .collect();
        for row_index in 0..height as usize {
            for column_index in 0..width as usize {
                let start = boundaries[row_index];
                let end = boundaries[row_index + 1];
                let occupied = (end - start.max((column_index + 1) as f64)).max(0.0);
                let expected = (255.0 * occupied / (end - start)).floor() as u8;
                assert!(
                    layers[0][row_index * width as usize + column_index].abs_diff(expected) <= 1
                );
            }
        }
    }

    #[test]
    fn boundaries_remain_finite_and_monotone_for_large_times() {
        for total in [0, 1, 2, 100, 1u64 << 53, u64::MAX] {
            for scale in [
                TimeScale::Linear,
                TimeScale::SquareRoot,
                TimeScale::Logarithmic,
            ] {
                let mut previous = 0.0;
                for row_index in 0..=1000 {
                    let boundary = scale.boundary(row_index as f64 / 1000.0, total as f64, 1.0);
                    assert!(
                        boundary.is_finite() && boundary >= previous && boundary <= total as f64
                    );
                    previous = boundary;
                }
                assert_eq!(previous, total as f64);
            }
        }
    }

    #[test]
    fn reconstruct_near_u64_limit_without_endpoint_overflow() {
        use crate::PowerOfTwo;
        let mut machine = PyramidMachine::new("1RA1RA", 2, 1, "linear-b").unwrap();
        let layer = machine
            .active
            .space_time_layers
            .values_mut()
            .next()
            .unwrap();
        layer.vis_step = u64::MAX - 1;
        layer.y_stride = PowerOfTwo::MAX;
        let mut time = 1u64 << 63;
        for exponent in (0..63).rev() {
            let mut line = layer.spacelines.main[0].clone();
            line.time = time;
            let weight = PowerOfTwo::from_exp(exponent);
            layer.spacelines.buffer0.push((line, weight));
            time += weight.as_u64();
        }
        assert_eq!(time, u64::MAX);
        for scale in [
            TimeScale::Linear,
            TimeScale::SquareRoot,
            TimeScale::Logarithmic,
        ] {
            let (_, _, pixels) = packed(&machine, scale);
            assert!(
                pixels
                    .iter()
                    .all(|layer| layer.iter().all(|pixel| *pixel == 0))
            );
        }
    }

    #[test]
    fn stopping_palette_changes_and_restart() {
        let mut machine = PyramidMachine::new(BB5_CHAMP, 16, 8, "linear-b").unwrap();
        assert!(!machine.step_for_secs(10.0, Some(137), 1000));
        assert_eq!(machine.step_count(), 137);
        let checkpoints = machine.checkpoint_count();
        for scale in ["sqrt", "log", "linear-b"] {
            machine.set_time_scale(scale).unwrap();
            assert_ne!(
                machine.to_png(&[255, 255, 255, 255, 0, 0]).unwrap(),
                machine.to_png(&[0, 0, 0, 0, 0, 255]).unwrap()
            );
            assert_eq!(machine.checkpoint_count(), checkpoints);
            assert_eq!(machine.step_count(), 137);
        }
        let fresh = PyramidMachine::new(BB5_CHAMP, 16, 8, "log").unwrap();
        assert_eq!(fresh.step_count(), 1);
        assert_eq!(fresh.checkpoint_count(), 1);
    }
}

#[cfg(test)]
mod spatial_tests;
