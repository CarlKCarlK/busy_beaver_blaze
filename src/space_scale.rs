//! Signed tape-coordinate maps and original-coordinate spatial integration.
use aligned_vec::AVec;

use crate::{ALIGN, PixelPolicy, PowerOfTwo, SpaceByTimeMachine, Spaceline};

/// Independent horizontal coordinate transformation; exponential uses a global extent normalization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpaceScale {
    /// Keep the reference renderer's horizontal bins exactly.
    #[default]
    Linear,
    /// Signed square root of the distance from tape zero.
    SquareRoot,
    /// Signed ln(1 + distance) from tape zero.
    Logarithmic,
    /// Signed square: widen tape regions farther from zero.
    Square,
    /// Signed exponential, normalized to the global extent when rendering.
    Exponential,
    /// Stretch each row linearly around tape zero according to its screen Y fraction.
    RowStretch,
}

impl SpaceScale {
    pub(crate) fn parse(scale: &str) -> Result<Self, String> {
        match scale {
            "linear" => Ok(Self::Linear),
            "sqrt" => Ok(Self::SquareRoot),
            "log" => Ok(Self::Logarithmic),
            "square" => Ok(Self::Square),
            "exp" => Ok(Self::Exponential),
            "row" => Ok(Self::RowStretch),
            _ => Err(format!("Unknown space scale: {scale}")),
        }
    }

    fn forward(self, coordinate: f64, unit: f64) -> f64 {
        assert!(unit.is_finite() && unit > 0.0);
        let distance = coordinate.abs() / unit;
        coordinate.signum()
            * unit
            * match self {
                Self::Linear => distance,
                Self::SquareRoot => distance.sqrt(),
                Self::Logarithmic => distance.ln_1p(),
                Self::Square => distance * distance,
                Self::Exponential => distance.exp_m1(),
                Self::RowStretch => unreachable!("row stretch requires a screen row"),
            }
    }

    fn inverse(self, coordinate: f64, unit: f64) -> f64 {
        assert!(unit.is_finite() && unit > 0.0);
        let distance = coordinate.abs() / unit;
        coordinate.signum()
            * unit
            * match self {
                Self::Linear => distance,
                Self::SquareRoot => distance * distance,
                Self::Logarithmic => distance.exp_m1(),
                Self::Square => distance.sqrt(),
                Self::Exponential => distance.ln_1p(),
                Self::RowStretch => unreachable!("row stretch requires a screen row"),
            }
    }
}

/// Tape-coordinate intervals including the reference image's blank edge padding.
/// Global maps use fixed intervals; row stretch scales them about the fixed origin.
/// Neither mode normalizes to each row's occupied tape width.
pub(crate) struct SpatialMap {
    pub(crate) intervals: Vec<(f64, f64)>,
    representatives: Vec<f64>,
}

impl SpatialMap {
    pub(crate) fn new(
        scale: SpaceScale,
        negative_len: usize,
        nonnegative_len: usize,
        stride: PowerOfTwo,
    ) -> Self {
        let width = negative_len + nonnegative_len;
        assert!(width > 0);
        let stride = stride.as_u64() as f64;
        let lower = -(negative_len as f64) * stride;
        let upper = nonnegative_len as f64 * stride;
        // Match the inverse of the normalized logarithmic map. The exponent
        // stays <= ln(1 + extent), even for tapes too wide for exp(tape_index).
        let extent = lower.abs().max(upper);
        let unit = if scale == SpaceScale::Exponential {
            extent / extent.ln_1p()
        } else {
            1.0
        };
        let transformed_lower = scale.forward(lower, unit);
        let transformed_upper = scale.forward(upper, unit);
        let transformed = |position: f64| {
            transformed_lower + (transformed_upper - transformed_lower) * position / width as f64
        };
        let boundary = |column: usize| {
            if column == 0 {
                lower
            } else if column == width {
                upper
            } else {
                scale.inverse(transformed(column as f64), unit)
            }
        };
        let mut intervals = Vec::with_capacity(width);
        let mut representatives = Vec::with_capacity(width);
        for column_index in 0..width {
            let start = boundary(column_index);
            let end = boundary(column_index + 1);
            assert!(start.is_finite() && end.is_finite() && end > start);
            intervals.push((start, end));
            representatives.push(scale.inverse(transformed(column_index as f64 + 0.5), unit));
        }
        Self {
            intervals,
            representatives,
        }
    }

    fn stretch_for_row(
        &mut self,
        negative_len: usize,
        stride: PowerOfTwo,
        row_index: usize,
        height: usize,
    ) {
        assert!(height > 0 && row_index < height);
        // Finite at the top, exactly 1x at the bottom; one-row images stay unchanged.
        let fraction = if height == 1 {
            1.0
        } else {
            (row_index as f64 / (height - 1) as f64).max(1.0 / height as f64)
        };
        let cell_width = stride.as_u64() as f64 * fraction;
        for (column_index, (interval, representative)) in self
            .intervals
            .iter_mut()
            .zip(&mut self.representatives)
            .enumerate()
        {
            let start = (column_index as f64 - negative_len as f64) * cell_width;
            *interval = (start, start + cell_width);
            *representative = start + 0.5 * cell_width;
        }
    }

    pub(crate) fn accumulate(&self, source: &SpatialRow, weight: f64, sums: &mut [f64]) {
        assert_eq!(sums.len(), self.intervals.len());
        for (sum, &(start, end)) in sums.iter_mut().zip(&self.intervals) {
            *sum += source.average(start, end) * weight;
        }
    }

    fn render(&self, source: &SpatialRow, policy: PixelPolicy, output: &mut [u8]) {
        for (column_index, pixel) in output.iter_mut().enumerate() {
            *pixel = match policy {
                PixelPolicy::Binning => {
                    let (start, end) = self.intervals[column_index];
                    quantize(source.average(start, end))
                }
                PixelPolicy::Sampling => source.sample(self.representatives[column_index]),
            };
        }
    }
}

/// Prefix integrals grow away from zero on each tape half. Keeping the halves
/// separate avoids subtracting enormous tape offsets when magnifying the origin.
pub(crate) struct SpatialRow {
    negative: Vec<f64>,
    nonnegative: Vec<f64>,
    stride: f64,
}

impl Default for SpatialRow {
    fn default() -> Self {
        Self {
            negative: vec![0.0],
            nonnegative: vec![0.0],
            stride: 1.0,
        }
    }
}

impl SpatialRow {
    fn fill_prefix(prefix: &mut Vec<f64>, values: impl Iterator<Item = u8>, length: usize) {
        prefix.clear();
        prefix.reserve(length + 1);
        prefix.push(0.0);
        for value in values {
            prefix.push(prefix.last().unwrap() + f64::from(value));
        }
    }

    pub(crate) fn replace_line(&mut self, line: &Spaceline) {
        Self::fill_prefix(
            &mut self.negative,
            line.negative.iter().map(|pixel| pixel.as_u8()),
            line.negative.len(),
        );
        Self::fill_prefix(
            &mut self.nonnegative,
            line.nonnegative.iter().map(|pixel| pixel.as_u8()),
            line.nonnegative.len(),
        );
        self.stride = line.x_stride.as_u64() as f64;
    }

    fn replace_packed(&mut self, row: &[u8], negative_len: usize, stride: PowerOfTwo) {
        let (negative, nonnegative) = row.split_at(negative_len);
        Self::fill_prefix(
            &mut self.negative,
            negative.iter().rev().copied(),
            negative.len(),
        );
        Self::fill_prefix(
            &mut self.nonnegative,
            nonnegative.iter().copied(),
            nonnegative.len(),
        );
        self.stride = stride.as_u64() as f64;
    }

    #[cfg(test)]
    fn from_packed(row: &[u8], negative_len: usize, stride: PowerOfTwo) -> Self {
        let mut source = Self::default();
        source.replace_packed(row, negative_len, stride);
        source
    }

    fn integral_half(&self, prefix: &[f64], distance: f64) -> f64 {
        let position = distance / self.stride;
        let whole = position.floor() as usize;
        if whole >= prefix.len() - 1 {
            return prefix.last().unwrap() * self.stride;
        }
        (prefix[whole] + (prefix[whole + 1] - prefix[whole]) * (position - whole as f64))
            * self.stride
    }

    fn integral(&self, coordinate: f64) -> f64 {
        if coordinate < 0.0 {
            -self.integral_half(&self.negative, -coordinate)
        } else {
            self.integral_half(&self.nonnegative, coordinate)
        }
    }

    fn average(&self, start: f64, end: f64) -> f64 {
        assert!(end > start);
        (self.integral(end) - self.integral(start)) / (end - start)
    }

    fn sample(&self, coordinate: f64) -> u8 {
        let (prefix, index) = if coordinate < 0.0 {
            let count = (-coordinate / self.stride).ceil() as usize;
            assert!(count > 0);
            (&self.negative, count - 1)
        } else {
            (
                &self.nonnegative,
                (coordinate / self.stride).floor() as usize,
            )
        };
        if index >= prefix.len() - 1 {
            return 0;
        }
        (prefix[index + 1] - prefix[index]) as u8
    }
}

pub(crate) fn quantize(value: f64) -> u8 {
    assert!(
        value.is_finite() && (-1e-6..=255.0 + 1e-6).contains(&value),
        "invalid layer average {value}"
    );
    // Clamp only floating-point endpoint error; invalid averages fail above.
    value.clamp(0.0, 255.0).floor() as u8
}

/// Preserve Linear A's temporal packing (including final padding and Sampling),
/// then transform each native symbol layer before palette composition.
pub(crate) fn linear_spatial_data(
    machine: &mut SpaceByTimeMachine,
    scale: SpaceScale,
) -> Result<(u32, u32, Vec<AVec<u8>>), String> {
    let tape = machine.machine.tape();
    let (negative_len, nonnegative_len) = (tape.negative.len(), tape.nonnegative.len());
    spatial_data(
        &mut machine.space_time_layers,
        negative_len,
        nonnegative_len,
        scale,
    )
}

/// Resample packed symbol layers after temporal packing. Row stretch therefore
/// depends on final screen rows, including when temporal packing halves the height.
pub(crate) fn spatial_data(
    layers: &mut crate::space_time_layers::SpaceTimeLayers,
    tape_negative_len: usize,
    tape_nonnegative_len: usize,
    scale: SpaceScale,
) -> Result<(u32, u32, Vec<AVec<u8>>), String> {
    let first = layers.first();
    let (x_goal, y_goal) = (first.x_goal as usize, first.y_goal as usize);
    let stride = Spaceline::find_x_stride(tape_negative_len, tape_nonnegative_len, x_goal);
    let negative_len = stride.div_ceil_into(tape_negative_len);
    let nonnegative_len = stride.div_ceil_into(tape_nonnegative_len);
    let width = negative_len + nonnegative_len;
    let map_scale = if scale == SpaceScale::RowStretch {
        SpaceScale::Linear
    } else {
        scale
    };
    let mut map = SpatialMap::new(map_scale, negative_len, nonnegative_len, stride);
    let symbol_count =
        u8::try_from(layers.values().count()).expect("symbol layer count must fit u8");
    let mut output = Vec::new();
    let mut dimensions = None;
    let mut source = SpatialRow::default();
    let mut transformed: AVec<u8> = AVec::with_capacity(ALIGN, width);
    transformed.resize(width, 0);
    // Index layers only once per frame; keep a new mutable-iterator caller out of
    // the reference machine's native iterator monomorphization.
    for select in 1..=symbol_count {
        let select = core::num::NonZeroU8::new(select).unwrap();
        let layer = layers.get_mut(select).unwrap();
        let (actual_width, actual_height, mut packed) = layer
            .to_packed_data(tape_negative_len, tape_nonnegative_len, x_goal, y_goal)
            .map_err(|error| error.to_string())?;
        assert_eq!(actual_width as usize, width);
        if let Some(dimensions) = dimensions {
            assert_eq!(dimensions, (actual_width, actual_height));
        }
        dimensions = Some((actual_width, actual_height));
        for (row_index, row) in packed.chunks_exact_mut(width).enumerate() {
            if scale == SpaceScale::RowStretch {
                map.stretch_for_row(negative_len, stride, row_index, actual_height as usize);
            }
            source.replace_packed(row, negative_len, stride);
            map.render(&source, layer.pixel_policy, &mut transformed);
            row.copy_from_slice(&transformed);
        }
        output.push(packed);
    }
    let (width, height) = dimensions.expect("symbol layers must exist");
    Ok((width, height, output))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn signed_maps_and_inverses() {
        close(SpaceScale::SquareRoot.forward(9.0, 1.0), 3.0);
        close(SpaceScale::SquareRoot.forward(-4.0, 1.0), -2.0);
        close(SpaceScale::Logarithmic.forward(1.0, 1.0), 2f64.ln());
        close(SpaceScale::Logarithmic.forward(-3.0, 1.0), -4f64.ln());
        for scale in [
            SpaceScale::Linear,
            SpaceScale::SquareRoot,
            SpaceScale::Logarithmic,
        ] {
            for coordinate in [0.0, 1e-12, 0.25, 1.0, 9.0, 1e9, u64::MAX as f64] {
                for unit in [1.0, 2.0] {
                    let mapped = scale.forward(coordinate, unit);
                    assert!(mapped.is_finite());
                    close(scale.forward(-coordinate, unit), -mapped);
                    close(scale.inverse(mapped, unit), coordinate);
                    close(scale.inverse(-mapped, unit), -coordinate);
                }
            }
        }
    }

    #[test]
    fn origin_and_order_across_strides() {
        for stride in [
            PowerOfTwo::ONE,
            PowerOfTwo::from_exp(10),
            PowerOfTwo::from_exp(60),
        ] {
            for scale in [
                SpaceScale::Linear,
                SpaceScale::SquareRoot,
                SpaceScale::Logarithmic,
            ] {
                let map = SpatialMap::new(scale, 3, 3, stride);
                for column_index in 0..6 {
                    let (start, end) = map.intervals[column_index];
                    assert!(end > start);
                    let (opposite_start, opposite_end) = map.intervals[5 - column_index];
                    close(start, -opposite_end);
                    close(end, -opposite_start);
                }
                close(map.intervals[2].1, 0.0);
                close(map.intervals[3].0, 0.0);
            }
        }
        // The asymmetric [-9,16] range maps sqrt endpoints to [-3,4].
        // Zero lies 3/7 across the *global* image, independent of row contents.
        let map = SpatialMap::new(SpaceScale::SquareRoot, 9, 16, PowerOfTwo::ONE);
        close(map.intervals[10].0, -0.04);
        close(map.intervals[10].1, 0.0064);
    }

    #[test]
    fn weighted_spatial_average_matches_direct_cell_overlaps() {
        let values = [255, 0, 77, 180, 0, 255, 33];
        for stride in [PowerOfTwo::ONE, PowerOfTwo::EIGHT, PowerOfTwo::from_exp(40)] {
            for scale in [
                SpaceScale::Linear,
                SpaceScale::SquareRoot,
                SpaceScale::Logarithmic,
            ] {
                let map = SpatialMap::new(scale, 3, 4, stride);
                let source = SpatialRow::from_packed(&values, 3, stride);
                for &(start, end) in &map.intervals {
                    let mut area = 0.0;
                    for (column_index, value) in values.iter().enumerate() {
                        let cell_start = (column_index as f64 - 3.0) * stride.as_u64() as f64;
                        let cell_end = cell_start + stride.as_u64() as f64;
                        area += (end.min(cell_end) - start.max(cell_start)).max(0.0)
                            * f64::from(*value);
                    }
                    close(source.average(start, end), area / (end - start));
                }
            }
        }
    }

    #[test]
    fn magnified_origin_and_blank_extent_keep_correct_weights() {
        let source = SpatialRow::from_packed(&[255, 255], 1, PowerOfTwo::from_exp(60));
        close(source.average(-1e-6, 1e-6), 255.0);
        let source = SpatialRow::from_packed(&[255, 255], 1, PowerOfTwo::ONE);
        close(source.average(-2.0, 2.0), 127.5);
        assert_eq!(source.sample(-1.0), 255);
        assert_eq!(source.sample(1.0), 0);
    }

    #[test]
    fn speed_uses_representative_samples() {
        let values = [0, 255, 77, 180, 0, 33, 255];
        let source = SpatialRow::from_packed(&values, 3, PowerOfTwo::ONE);
        for scale in [SpaceScale::SquareRoot, SpaceScale::Logarithmic] {
            let map = SpatialMap::new(scale, 3, 4, PowerOfTwo::ONE);
            let mut output = [0; 7];
            map.render(&source, PixelPolicy::Sampling, &mut output);
            for value in output {
                assert!(values.contains(&value));
            }
            assert_ne!(output, values);
        }
    }
}

#[cfg(test)]
mod outward_tests {
    use super::*;

    #[test]
    fn square_and_exponential_widen_outer_columns_without_overflow() {
        assert_eq!(SpaceScale::Square.forward(-3.0, 1.0), -9.0);
        assert_eq!(SpaceScale::Square.inverse(-9.0, 1.0), -3.0);
        for scale in [SpaceScale::Square, SpaceScale::Exponential] {
            for stride in [PowerOfTwo::ONE, PowerOfTwo::from_exp(60)] {
                let map = SpatialMap::new(scale, 4, 4, stride);
                for column in 0..8 {
                    let (start, end) = map.intervals[column];
                    let (opposite_start, opposite_end) = map.intervals[7 - column];
                    assert!(start.is_finite() && end.is_finite() && end > start);
                    assert!((start + opposite_end).abs() <= 1e-12 * start.abs().max(1.0));
                    assert!((end + opposite_start).abs() <= 1e-12 * end.abs().max(1.0));
                }
                let inner = map.intervals[4];
                let outer = map.intervals[7];
                assert!(inner.1 - inner.0 > outer.1 - outer.0);
            }
        }
    }

    #[test]
    fn normalized_inverse_boundaries_match_independent_formulas() {
        let values = [0, 33, 77, 180, 255, 120, 0, 255];
        let source = SpatialRow::from_packed(&values, 0, PowerOfTwo::ONE);
        for scale in [SpaceScale::Square, SpaceScale::Exponential] {
            let map = SpatialMap::new(scale, 0, 8, PowerOfTwo::ONE);
            let boundary = |column: usize| {
                let position = column as f64 / 8.0;
                match scale {
                    SpaceScale::Square => 8.0 * position.sqrt(),
                    SpaceScale::Exponential => 8.0 * (1.0 + 8.0 * position).ln() / 9f64.ln(),
                    _ => unreachable!(),
                }
            };
            for column in 0..8 {
                let (start, end) = map.intervals[column];
                assert!((start - boundary(column)).abs() < 1e-12);
                assert!((end - boundary(column + 1)).abs() < 1e-12);
                let area: f64 = values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        (end.min(index as f64 + 1.0) - start.max(index as f64)).max(0.0)
                            * f64::from(*value)
                    })
                    .sum();
                assert!((source.average(start, end) - area / (end - start)).abs() < 1e-10);
            }
        }
    }
}

#[cfg(test)]
mod row_tests {
    use super::*;

    #[test]
    fn row_stretch_is_linear_origin_anchored_and_finite() {
        let mut map = SpatialMap::new(SpaceScale::Linear, 4, 4, PowerOfTwo::ONE);
        for row_index in 0..5 {
            map.stretch_for_row(4, PowerOfTwo::ONE, row_index, 5);
            let width = map.intervals[0].1 - map.intervals[0].0;
            assert!(width > 0.0 && width.is_finite());
            assert_eq!(map.intervals[4].0, 0.0);
            for &(start, end) in &map.intervals {
                assert!((end - start - width).abs() < 1e-12);
            }
        }
        assert_eq!(map.intervals[0], (-4.0, -3.0)); // Bottom row is exactly 1x.
        map.stretch_for_row(4, PowerOfTwo::ONE, 0, 1);
        assert_eq!(map.intervals[0], (-4.0, -3.0)); // Single-row images stay unchanged.
    }

    #[test]
    fn row_stretch_removes_linear_widening_in_both_pixel_policies() {
        for policy in [PixelPolicy::Binning, PixelPolicy::Sampling] {
            let mut map = SpatialMap::new(SpaceScale::Linear, 0, 4, PowerOfTwo::ONE);
            for row_index in 1..5 {
                let values: Vec<u8> = (0..4)
                    .map(|column| if column < row_index { 255 } else { 0 })
                    .collect();
                let source = SpatialRow::from_packed(&values, 0, PowerOfTwo::ONE);
                map.stretch_for_row(0, PowerOfTwo::ONE, row_index, 5);
                let mut output = [0u8; 4];
                map.render(&source, policy, &mut output);
                assert_eq!(output, [255; 4]);
            }
        }
    }
}
