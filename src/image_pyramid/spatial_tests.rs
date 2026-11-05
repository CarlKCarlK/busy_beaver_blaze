use super::*;
use crate::space_scale::linear_spatial_data;
use crate::{BB5_CHAMP, Machine};

fn packed(
    machine: &PyramidMachine,
    time_scale: TimeScale,
    space_scale: SpaceScale,
) -> (usize, usize, Vec<AVec<u8>>) {
    let mut layers =
        machine
            .pyramid
            .reconstruct_with_space(&machine.active, time_scale, space_scale);
    let first = machine.active.space_time_layers.first();
    let (_, width, height, pixels) = layers
        .png_data_and_packed_data(
            &[[255, 255, 255], [0, 0, 0]],
            machine.active.machine.tape.negative.len(),
            machine.active.machine.tape.nonnegative.len(),
            (first.x_goal as usize, first.y_goal as usize),
        )
        .unwrap();
    (width as usize, height as usize, pixels)
}

// Deliberately independent of the production mapping/integral helpers.
fn forward(coordinate: f64, scale: SpaceScale) -> f64 {
    match scale {
        SpaceScale::Linear => coordinate,
        SpaceScale::SquareRoot => coordinate.abs().sqrt().copysign(coordinate),
        SpaceScale::Logarithmic => (1.0 + coordinate.abs()).ln().copysign(coordinate),
    }
}
fn inverse(coordinate: f64, scale: SpaceScale) -> f64 {
    match scale {
        SpaceScale::Linear => coordinate,
        SpaceScale::SquareRoot => coordinate.powi(2).copysign(coordinate),
        SpaceScale::Logarithmic => (coordinate.abs().exp() - 1.0).copysign(coordinate),
    }
}

fn reference(
    machine: &PyramidMachine,
    history: &[Vec<u8>],
    tape_origin: i64,
    time_scale: TimeScale,
    space_scale: SpaceScale,
) -> Vec<Vec<u8>> {
    let first = machine.active.space_time_layers.first();
    let tape = machine.active.machine.tape();
    let stride = Spaceline::find_x_stride(
        tape.negative.len(),
        tape.nonnegative.len(),
        first.x_goal as usize,
    );
    let lower = -(stride.div_ceil_into(tape.negative.len()) as f64) * stride.as_u64() as f64;
    let upper = stride.div_ceil_into(tape.nonnegative.len()) as f64 * stride.as_u64() as f64;
    let width =
        stride.div_ceil_into(tape.negative.len()) + stride.div_ceil_into(tape.nonnegative.len());
    let internal_height = first.spacelines.len();
    let height = if time_scale != TimeScale::Linear && internal_height == 2 * first.y_goal as usize
    {
        first.y_goal as usize
    } else {
        internal_height
    };
    let total = history.len() as f64;
    let boundary = |row_index: usize| match time_scale {
        TimeScale::Linear => row_index as f64 * first.y_stride.as_u64() as f64,
        TimeScale::SquareRoot => total * (row_index as f64 / height as f64).powi(2),
        TimeScale::Logarithmic => (1.0 + total).powf(row_index as f64 / height as f64) - 1.0,
    };
    let mut output = Vec::new();
    for select in 1..machine.active.machine.symbol_count() {
        let mut pixels = Vec::new();
        for row_index in 0..height {
            let (start_time, end_time) = (boundary(row_index), boundary(row_index + 1));
            for column_index in 0..width {
                let start_x = inverse(
                    forward(lower, space_scale)
                        + (forward(upper, space_scale) - forward(lower, space_scale))
                            * column_index as f64
                            / width as f64,
                    space_scale,
                );
                let end_x = inverse(
                    forward(lower, space_scale)
                        + (forward(upper, space_scale) - forward(lower, space_scale))
                            * (column_index + 1) as f64
                            / width as f64,
                    space_scale,
                );
                let mut area = 0.0;
                for (step_index, row) in history
                    .iter()
                    .enumerate()
                    .skip(start_time.floor() as usize)
                    .take((end_time.ceil() - start_time.floor()) as usize)
                {
                    let time_weight = (end_time.min(step_index as f64 + 1.0)
                        - start_time.max(step_index as f64))
                    .max(0.0);
                    for tape_index in start_x.floor() as i64..end_x.ceil() as i64 {
                        let space_weight = (end_x.min(tape_index as f64 + 1.0)
                            - start_x.max(tape_index as f64))
                        .max(0.0);
                        let index = tape_index - tape_origin;
                        let symbol = if index >= 0 {
                            row.get(index as usize).copied().unwrap_or(0)
                        } else {
                            0
                        };
                        if symbol == select {
                            area += 255.0 * time_weight * space_weight;
                        }
                    }
                }
                pixels.push(
                    (area / ((end_x - start_x) * (end_time - start_time)))
                        .clamp(0.0, 255.0)
                        .floor() as u8,
                );
            }
        }
        if time_scale == TimeScale::Linear && internal_height == 2 * first.y_goal as usize {
            pixels = pixels
                .chunks_exact(width * 2)
                .flat_map(|rows| {
                    rows[..width]
                        .iter()
                        .zip(&rows[width..])
                        .map(|(first, second)| crate::Pixel::mean_bytes(*first, *second))
                })
                .collect();
        }
        output.push(pixels);
    }
    output
}

#[test]
fn independent_rectangle_reference_report() {
    let mut metrics = [[(0u8, 0u64, 0u64); 9]; 2];
    for program in [BB5_CHAMP, "1RB2LA3RA0LB_3LA2RB1LB0RA"] {
        for transitions in [0, 1, 7, 31, 257] {
            for (x_goal, y_goal) in [(512, 512), (8, 5)] {
                let mut machine = PyramidMachine::new(program, x_goal, y_goal, "linear-b").unwrap();
                let mut original = Machine::from_string(program).unwrap();
                let tape_origin = -(transitions + 2);
                let mut history = vec![
                    (tape_origin..-tape_origin)
                        .map(|tape_index| u8::from(original.tape().read(tape_index)))
                        .collect::<Vec<_>>(),
                ];
                for _step_index in 0..transitions {
                    let continues = machine.step();
                    assert_eq!(original.next().is_some(), continues);
                    if !continues {
                        break;
                    }
                    history.push(
                        (tape_origin..-tape_origin)
                            .map(|tape_index| u8::from(original.tape().read(tape_index)))
                            .collect(),
                    );
                }
                for (time_index, time_scale) in [
                    TimeScale::Linear,
                    TimeScale::SquareRoot,
                    TimeScale::Logarithmic,
                ]
                .into_iter()
                .enumerate()
                {
                    for (space_index, space_scale) in [
                        SpaceScale::Linear,
                        SpaceScale::SquareRoot,
                        SpaceScale::Logarithmic,
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let (_, _, actual) = packed(&machine, time_scale, space_scale);
                        let expected =
                            reference(&machine, &history, tape_origin, time_scale, space_scale);
                        let metric =
                            &mut metrics[usize::from(x_goal != 512)][time_index * 3 + space_index];
                        for (actual, expected) in actual.iter().zip(expected) {
                            assert_eq!(actual.len(), expected.len());
                            for (actual, expected) in actual.iter().zip(expected) {
                                let difference = actual.abs_diff(expected);
                                if x_goal == 512 {
                                    assert!(
                                        difference <= 1,
                                        "uncompressed error {difference}, time={time_scale:?} space={space_scale:?}"
                                    );
                                }
                                metric.0 = metric.0.max(difference);
                                metric.1 += u64::from(difference);
                                metric.2 += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    for (compressed, metrics) in metrics.into_iter().enumerate() {
        for (index, (maximum, sum, count)) in metrics.into_iter().enumerate() {
            println!(
                "Direct rectangle reference compressed={compressed} Y={} X={}: max={maximum} mean={:.6} pixels={count}",
                index / 3,
                index % 3,
                sum as f64 / count as f64
            );
            if compressed == 0 {
                assert!(maximum <= 1);
                assert!(sum as f64 / (count as f64) < 0.13);
            } else {
                // Quantify lost source detail separately from mapping correctness.
                let limits = [
                    (1, 0.01),
                    (155, 3.8),
                    (148, 5.5),
                    (47, 0.9),
                    (181, 4.95),
                    (169, 6.95),
                    (21, 0.45),
                    (200, 4.95),
                    (192, 6.95),
                ];
                assert!(maximum <= limits[index].0);
                assert!(sum as f64 / (count as f64) < limits[index].1);
            }
        }
    }
}

#[test]
fn linear_a_b_spatial_comparison_and_checkpoint_invariance() {
    let mut maximum = 0u8;
    let mut sum = 0u64;
    let mut count = 0u64;
    for program in [BB5_CHAMP, "1RB2LA3RA0LB_3LA2RB1LB0RA"] {
        let mut machine = PyramidMachine::new(program, 16, 9, "linear-b").unwrap();
        let mut reference = SpaceByTimeMachine::from_str(program, 16, 9, true, 0).unwrap();
        for step_index in 0u64..=10_001 {
            if step_index < 4
                || step_index.is_power_of_two()
                || (step_index + 1).is_power_of_two()
                || (step_index > 0 && (step_index - 1).is_power_of_two())
                || step_index == 10_001
            {
                let checkpoints = machine.checkpoint_count();
                let retained = machine.retained_bytes();
                for space_scale in [SpaceScale::SquareRoot, SpaceScale::Logarithmic] {
                    let (width, height, expected) =
                        linear_spatial_data(&mut reference, space_scale).unwrap();
                    let (actual_width, actual_height, actual) =
                        packed(&machine, TimeScale::Linear, space_scale);
                    assert_eq!(
                        (width as usize, height as usize),
                        (actual_width, actual_height)
                    );
                    for (expected, actual) in expected
                        .iter()
                        .flat_map(|layer| layer.iter())
                        .zip(actual.iter().flat_map(|layer| layer.iter()))
                    {
                        let difference = expected.abs_diff(*actual);
                        maximum = maximum.max(difference);
                        sum += u64::from(difference);
                        count += 1;
                    }
                    assert_eq!(machine.checkpoint_count(), checkpoints);
                    assert_eq!(machine.retained_bytes(), retained);
                }
            }
            if !machine.step() {
                break;
            }
            assert!(reference.next().is_some());
        }
    }
    println!(
        "Spatial Linear A/B comparison: max={maximum} mean={:.6} pixels={count}",
        sum as f64 / count as f64
    );
    assert!(maximum <= 4);
    assert!(sum as f64 / (count as f64) < 0.05);
}

#[test]
fn actual_resolution_changes_keep_axes_aligned() {
    let mut machine = PyramidMachine::new(BB5_CHAMP, 4, 3, "linear-b").unwrap();
    let mut reference = SpaceByTimeMachine::from_str(BB5_CHAMP, 4, 3, true, 0).unwrap();
    let mut previous = None;
    let mut horizontal_changes = 0;
    let mut vertical_changes = 0;
    for _step_index in 0..1025 {
        let tape = machine.active.machine.tape();
        let strides = (
            Spaceline::find_x_stride(tape.negative.len(), tape.nonnegative.len(), 4),
            machine.active.space_time_layers.first().y_stride,
        );
        if let Some((x_stride, y_stride)) = previous {
            horizontal_changes += usize::from(strides.0 != x_stride);
            vertical_changes += usize::from(strides.1 != y_stride);
        }
        previous = Some(strides);
        // Compare every step, so the immediate before/at/after states of actual
        // tape-width and time-stride changes are all covered, not just checkpoints.
        for space_scale in [
            SpaceScale::Linear,
            SpaceScale::SquareRoot,
            SpaceScale::Logarithmic,
        ] {
            let (width, height, expected) =
                linear_spatial_data(&mut reference, space_scale).unwrap();
            let (actual_width, actual_height, actual) =
                packed(&machine, TimeScale::Linear, space_scale);
            assert_eq!(
                (width as usize, height as usize),
                (actual_width, actual_height)
            );
            for (actual, expected) in actual
                .iter()
                .flat_map(|layer| layer.iter())
                .zip(expected.iter().flat_map(|layer| layer.iter()))
            {
                assert!(actual.abs_diff(*expected) <= 4);
            }
            for time_scale in [TimeScale::SquareRoot, TimeScale::Logarithmic] {
                let (actual_width, actual_height, layers) =
                    packed(&machine, time_scale, space_scale);
                assert_eq!(actual_width, width as usize);
                assert_eq!(actual_height, height as usize);
                assert!(
                    layers
                        .iter()
                        .all(|layer| layer.len() == actual_width * actual_height)
                );
            }
        }
        assert!(machine.step());
        assert!(reference.next().is_some());
    }
    assert!(horizontal_changes >= 2 && vertical_changes >= 2);
    println!(
        "Actual stride transitions: horizontal={horizontal_changes}, vertical={vertical_changes}"
    );
}
