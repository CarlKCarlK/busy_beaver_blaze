use crate::{
    ALIGN, BB5_CHAMP, BB6_CONTENDER, Error, LogStepIterator, Machine, PixelPolicy, PngDataIterator,
    PowerOfTwo, SpaceByTime, SpaceByTimeMachine, average_with_iterators, average_with_simd,
    bool_u8::BoolU8, find_x_stride, pixel::Pixel, spaceline::Spaceline,
    test_utils::compress_x_no_simd_binning,
};
use aligned_vec::AVec;
use core::simd::Simd;
use itertools::Itertools;
use rand::{Rng, SeedableRng};
use std::{
    collections::{HashMap, HashSet},
    fs,
};
use thousands::Separable;

#[allow(clippy::shadow_reuse, clippy::integer_division_remainder_used)]
#[test]
fn bb5_champ_space_by_time_native() -> Result<(), Error> {
    let mut machine: Machine = BB5_CHAMP.parse()?;

    let goal_x: u32 = 1000;
    let goal_y: u32 = 1000;
    let pixel_policy: PixelPolicy = PixelPolicy::Sampling;
    let mut sample_space_by_time = SpaceByTime::new0(goal_x, goal_y, pixel_policy);

    let early_stop = Some(10_500_000);
    // let early_stop = Some(1_000_000);
    let debug_interval = Some(1_000_000);

    while let Some(previous_tape_index) = machine
        .next()
        .filter(|_| early_stop.is_none_or(|stop| sample_space_by_time.step_index() + 1 < stop))
    {
        if debug_interval
            .is_none_or(|debug_interval| sample_space_by_time.step_index() % debug_interval == 0)
        {
            println!(
                "Step {}: {:?},\t{}",
                sample_space_by_time.step_index().separate_with_commas(),
                machine,
                machine.tape().index_range_to_string(-10..=10)
            );
        }

        sample_space_by_time.snapshot(&machine, previous_tape_index);
        // let _ = sample_space_by_time.to_png();
    }

    let png_data = sample_space_by_time.to_png(
        machine.tape.negative.len(),
        machine.tape.nonnegative.len(),
        goal_x as usize,
        goal_y as usize,
        [255, 255, 255], // white
        [255, 165, 0],   // orange
    )?;
    fs::write("tests/expected/test.png", &png_data).unwrap(); // TODO handle error

    println!(
        "Final: Steps {}: {:?}, #1's {}",
        sample_space_by_time.step_index().separate_with_commas(),
        machine,
        machine.count_ones()
    );

    if early_stop.is_none() {
        assert_eq!(sample_space_by_time.step_index(), 47_176_870);
        assert_eq!(machine.count_ones(), 4098);
        assert_eq!(machine.state(), 7);
        assert_eq!(machine.tape_index(), -12242);
    }

    Ok(())
}

#[allow(
    clippy::shadow_unrelated,
    clippy::cognitive_complexity,
    clippy::too_many_lines
)]
#[test]
fn test_average() {
    let values = AVec::from_iter(
        ALIGN,
        [
            BoolU8::FALSE,
            BoolU8::FALSE,
            BoolU8::FALSE,
            BoolU8::TRUE,
            BoolU8::TRUE,
            BoolU8::FALSE,
            BoolU8::TRUE,
            BoolU8::TRUE,
            BoolU8::TRUE,
        ]
        .iter()
        .copied(),
    );

    let step = PowerOfTwo::ONE;
    let bytes: &[u8] = &[0, 0, 0, 255, 255, 0, 255, 255, 255];
    let expected = AVec::<Pixel>::from_iter(ALIGN, bytes.iter().map(Pixel::from));

    let result = average_with_iterators(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<1>(&values, step);
    assert_eq!(result, expected);

    let step = PowerOfTwo::TWO;
    let bytes = &[0u8, 127, 127, 255, 127];
    let expected = AVec::<Pixel>::from_iter(ALIGN, bytes.iter().map(Pixel::from));

    let result = average_with_iterators(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<1>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<2>(&values, step);
    assert_eq!(result, expected);
    // Expected to panic
    // let result = average_with_simd::<4>(&values, step);
    // assert_eq!(result, expected);

    let step = PowerOfTwo::FOUR;
    let bytes = &[63u8, 191, 63];
    let expected = AVec::<Pixel>::from_iter(ALIGN, bytes.iter().map(Pixel::from));
    let result = average_with_iterators(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<1>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<2>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<4>(&values, step);
    assert_eq!(result, expected);

    let step = PowerOfTwo::EIGHT;
    let bytes = &[127u8, 31];
    let expected = AVec::<Pixel>::from_iter(ALIGN, bytes.iter().map(Pixel::from));
    let result = average_with_iterators(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<1>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<2>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<4>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<8>(&values, step);
    assert_eq!(result, expected);

    let step = PowerOfTwo::SIXTEEN;
    let bytes = &[79u8];
    let expected = AVec::<Pixel>::from_iter(ALIGN, bytes.iter().map(Pixel::from));
    let result = average_with_iterators(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<1>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<2>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<4>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<8>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<16>(&values, step);
    assert_eq!(result, expected);

    let step = PowerOfTwo::THIRTY_TWO;
    let bytes = &[39u8];
    let expected = AVec::<Pixel>::from_iter(ALIGN, bytes.iter().map(Pixel::from));
    let result = average_with_iterators(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<1>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<2>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<4>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<8>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<16>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<32>(&values, step);
    assert_eq!(result, expected);

    let step = PowerOfTwo::SIXTY_FOUR;
    let bytes = &[19u8];
    let expected = AVec::<Pixel>::from_iter(ALIGN, bytes.iter().map(Pixel::from));
    let result = average_with_iterators(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<1>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<2>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<4>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<8>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<16>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<32>(&values, step);
    assert_eq!(result, expected);
    let result = average_with_simd::<64>(&values, step);
    assert_eq!(result, expected);
}

#[test]
fn resample_simd() {
    for len in [0, 1, 2, 3, 5, 101, 111, 4001] {
        let seed = 0;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut pixels: AVec<Pixel, _> =
            AVec::from_iter(ALIGN, (0..len).map(|_| rng.random::<bool>().into()));

        let mut reference = pixels.clone();
        compress_x_no_simd_binning(&mut reference);
        Spaceline::compress_x_simd_binning(&mut pixels);
        assert_eq!(pixels, reference);
    }
}

#[test]
fn test_find_stride() {
    for tape_neg_len in [0usize, 1, 2, 3, 4, 5, 10, 101, 9999, 1_000_007] {
        for tape_non_neg_len in [1usize, 2, 3, 4, 5, 10, 101, 9999, 1_000_007] {
            let tape_len = tape_neg_len + tape_non_neg_len;
            for goal_x in [2, 3, 4, 5, 6, 7, 10, 100, 1000, 33_333] {
                let x_stride = find_x_stride(tape_neg_len, tape_non_neg_len, goal_x);
                let neg_len = x_stride.div_ceil_into(tape_neg_len);
                let non_neg_len = x_stride.div_ceil_into(tape_non_neg_len);
                let len = neg_len + non_neg_len;
                if tape_len < goal_x {
                    assert!(x_stride == PowerOfTwo::ONE);
                } else {
                    assert!(goal_x <= len && len < goal_x * 2);
                }
            }
        }
    }
}

#[allow(clippy::shadow_reuse, clippy::too_many_lines)]
#[test]
fn combo() {
    // make a HashMap from the strings "BB5_CHAMP", "BB6_CONTENDER" to the string constants
    let program_name_to_string =
        HashMap::from([("BB5_CHAMP", BB5_CHAMP), ("BB6_CONTENDER", BB6_CONTENDER)]);

    let exception_set = HashSet::from(["ignore"]);

    for early_stop in [2, 5u64, 6, 7, 300u64, 1_000_000u64] {
        for goal_x in [2, 30, 360] {
            for goal_y in [2, 3, 4, 30, 432] {
                for program_name in ["BB5_CHAMP", "BB6_CONTENDER"] {
                    for binning in [false, true] {
                        let program_string = program_name_to_string[program_name];
                        // println!("program_string: {program_string}");
                        let mut reference_machine = SpaceByTimeMachine::from_str(
                            program_string,
                            goal_x,
                            goal_y,
                            binning,
                            0,
                        )
                        .unwrap();
                        reference_machine.nth_js(early_stop - 2);
                        // println!(
                        //     "reference_machine: {:?} {:?}",
                        //     reference_machine.space_by_time.y_stride,
                        //     reference_machine.space_by_time.spacelines
                        // );
                        let (reference_png_data, ref_x, ref_y, reference_packed_data) =
                            reference_machine
                                .png_data_and_packed_data([255, 255, 255], [255, 165, 0]);
                        // println!("---------------");
                        for part_count in [1, 2, 5, 16] {
                            let key = format!(
                                "early_stop: {early_stop}, goal_x: {goal_x}, goal_y: {goal_y}, program_name: {program_name}, binning: {binning}, part_count: {part_count}"
                            );
                            println!("{key}");

                            let png_data_iterator = PngDataIterator::new(
                                early_stop,
                                part_count,
                                program_string,
                                goal_x,
                                goal_y,
                                [255, 255, 255], // white
                                [255, 165, 0],   // orange
                                binning,
                                &[0u64; 0],
                            );
                            let mut space_by_time_machine =
                                png_data_iterator.into_space_by_time_machine();
                            let (png_data, x, y, packed_data) = space_by_time_machine
                                .png_data_and_packed_data([255, 255, 255], [255, 165, 0]);

                            // Must be the same length and a given value can vary by no more than y_stride.log2() + x_stride.log2()
                            let last_spacetime = space_by_time_machine
                                .space_by_time
                                .spacelines
                                .main
                                .last()
                                .unwrap();
                            let max_diff = space_by_time_machine.space_by_time.y_stride.log2()
                                + last_spacetime.x_stride.log2();
                            // println!("max_diff: {max_diff}");

                            let mut ok = true;
                            ok = ok && packed_data.len() == reference_packed_data.len();
                            for (ref_val, val) in
                                reference_packed_data.iter().zip(packed_data.iter())
                            {
                                ok = ok && ref_val.abs_diff(*val) <= max_diff;
                            }

                            if !ok {
                                if exception_set.contains(key.as_str()) {
                                    println!("Skip: PNG data does not match for \"{key}\"");
                                    continue;
                                }
                                println!(
                                    "goal_x {goal_x}, goal_y {goal_y}, ref_x: {ref_x}, ref_y: {ref_y}, x, {x}, y: {y}"
                                );
                                let ref_file = "tests/expected/combo_parts_ref.png";
                                fs::write(ref_file, &reference_png_data).unwrap();
                                let test_file = "tests/expected/combo_parts_test.png";
                                fs::write(test_file, &png_data).unwrap();
                                panic!("PNG data does not match for \"{key}\"");
                            }
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::shadow_reuse, clippy::too_many_lines)]
#[test]
fn one() {
    // make a HashMap from the strings "BB5_CHAMP", "BB6_CONTENDER" to the string constants
    let program_name_to_string =
        HashMap::from([("BB5_CHAMP", BB5_CHAMP), ("BB6_CONTENDER", BB6_CONTENDER)]);

    let early_stop = 5;
    let goal_x = 2;
    let goal_y = 2;
    let program_name = "BB5_CHAMP";
    let binning = false;
    let part_count = 2;

    let program_string = program_name_to_string[program_name];

    let mut reference_machine =
        SpaceByTimeMachine::from_str(program_string, goal_x, goal_y, binning, 0).unwrap();
    reference_machine.nth_js(early_stop - 2);
    let (reference_png_data, ..) = reference_machine.png_data_and_packed_data(
        [255, 255, 255], // white
        [255, 165, 0],   // orange
    );

    let key = format!(
        "early_stop: {early_stop}, goal_x: {goal_x}, goal_y: {goal_y}, program_name: {program_name}, binning: {binning}, part_count: {part_count}"
    );
    println!("{key}");

    let (zero_color, one_color) = ([255, 255, 255], [255, 165, 0]); // white, orange

    let png_data_iterator = PngDataIterator::new(
        early_stop,
        part_count,
        program_string,
        goal_x,
        goal_y,
        zero_color,
        one_color,
        binning,
        &[0u64; 0],
    );
    let mut space_by_time_machine = png_data_iterator.into_space_by_time_machine();
    let (png_data, ..) = space_by_time_machine.png_data_and_packed_data(zero_color, one_color);

    // println!("goal_x {goal_x}, goal_y {goal_y}, ref_x, {ref_x}, ref_y: {ref_y}, x, {x}, y: {y}");
    // println!("ref_packed: {ref_packed:?}");
    // println!("packed_data: {packed_data:?}");
    let ref_file = "tests/expected/one_parts_ref.png";
    fs::write(ref_file, &reference_png_data).unwrap();
    let test_file = "tests/expected/one_parts_test.png";
    fs::write(test_file, &png_data).unwrap();
    assert!(
        (png_data == reference_png_data),
        "PNG data does not match for {key}"
    );
}

#[test]
fn test_log_step() {
    let log_step_iterator = LogStepIterator::new(10, 10);
    assert_eq!(
        vec![0, 0, 0, 1, 1, 2, 3, 4, 6, 9],
        log_step_iterator.collect::<Vec<_>>()
    );
    // println!(
    //     "log_step_iterator: {:?}",
    //     log_step_iterator.collect::<Vec<_>>()
    // );
}

#[test]
fn frames() {
    // let early_stop = 1_000_000_000;
    // let frame_count = 4;
    // let part_count = 16;
    // let goal_x: u32 = 1920;
    // let goal_y: u32 = 1080;
    // let binning = true;

    let frame_count = 10;
    let early_stop = 30;
    let part_count = 2;
    let goal_x: u32 = 360;
    let goal_y: u32 = 30;
    let binning = true;

    // let early_stop = 250_000_000;
    // let frame_count = 1000;
    // let part_count = 42;
    // let goal_x: u32 = 360;
    // let goal_y: u32 = 432;
    // let binning = true;

    // let early_stop = 10_000_000_000;
    // let frame_count = 2000;
    // let part_count = 32;
    // let goal_x: u32 = 1920;
    // let goal_y: u32 = 1080;
    // let binning = true;

    // let early_stop = 250_000_000_000;
    // let frame_count = 2000;
    // let part_count = 32;
    // let goal_x: u32 = 1920;
    // let goal_y: u32 = 1080;
    // let binning = true;

    // let early_stop = 10_000_000;
    // let part_count = 16;
    // let goal_x: u32 = 360;
    // let goal_y: u32 = 30;
    // let binning = true;

    // let early_stop = 1_000_000;
    // let part_count = 1;
    // let goal_x: u32 = 360;
    // let goal_y: u32 = 30;
    // let binning = true;

    // let early_stop = 300;
    // let part_count = 2;
    // let goal_x: u32 = 360;
    // let goal_y: u32 = 432;
    // let binning = true;

    // let early_stop = 5u64;
    // let part_count = 10;

    // let goal_x: u32 = 360;
    // let goal_y: u32 = 432;
    let frame_index_to_step_index: Vec<_> = LogStepIterator::new(early_stop, frame_count).collect();

    let png_data_iterator = PngDataIterator::new(
        early_stop,
        part_count,
        BB6_CONTENDER,
        goal_x,
        goal_y,
        [255, 255, 255], // white
        [255, 165, 0],   // orange
        binning,
        frame_index_to_step_index.as_slice(),
    );

    for (frame_index, (step_index, png_data)) in png_data_iterator.enumerate() {
        let cmk_file = format!(r"M:\deldir\bb\frames_test2\cmk{frame_index:07}.png");
        println!("Frame {}, Step {}", frame_index, step_index + 1);
        fs::write(cmk_file, &png_data).unwrap();
    }
}

#[test]
fn stop_early() {
    let early_stop = 47_176_869;
    let frame_count = 100;
    let part_count = 5;
    let goal_x: u32 = 1920;
    let goal_y: u32 = 1080;
    let zero_color = [255, 255, 255]; // white
    let one_color = [255, 165, 0]; // orange
    let binning = true;

    let frame_index_to_step_index = LogStepIterator::new(early_stop, frame_count).collect_vec();
    let png_data_iterator = PngDataIterator::new(
        early_stop,
        part_count,
        BB5_CHAMP,
        goal_x,
        goal_y,
        zero_color,
        one_color,
        binning,
        &frame_index_to_step_index,
    );

    let output_dir = r"M:\deldir\bb\stop_early";
    fs::create_dir_all(output_dir).unwrap();

    for (frame_index, (step_index, png_data)) in png_data_iterator.enumerate() {
        let cmk_file = format!(r"{output_dir}\cmk{frame_index:07}.png");
        println!("Frame {}, Step {}", frame_index, step_index + 1);
        fs::write(cmk_file, &png_data).unwrap();
    }
}

#[test]
#[allow(clippy::shadow_reuse)]
fn interleave() {
    let left = Simd::from_array([0, 1, 2, 3]);
    let right = Simd::from_array([4, 5, 6, 7]);
    let (left, right) = left.deinterleave(right);
    assert_eq!(left.to_array(), [0, 2, 4, 6]);
    assert_eq!(right.to_array(), [1, 3, 5, 7]);
}

#[test]
fn compress_x_simd_binning() {
    let mut pixels = AVec::from_iter(ALIGN, (0u8..201).map(Pixel::from));
    pixels[0] = Pixel::from(255u8);
    let mut reference = AVec::from_iter(ALIGN, (0u8..201).step_by(2).map(Pixel::from));
    reference[0] = Pixel::from(128u8);
    reference[100] = Pixel::from(100u8);
    Spaceline::compress_x_simd_binning(&mut pixels);
    println!("pixels: {pixels:?}");
    assert_eq!(pixels, reference);
}

#[test]
fn compress_x_simd_sampling() {
    let mut pixels = AVec::from_iter(ALIGN, (0u8..201).map(Pixel::from));
    pixels[0] = Pixel::from(255u8);
    let mut reference = AVec::from_iter(ALIGN, (0u8..201).step_by(2).map(Pixel::from));
    reference[0] = Pixel::from(255u8);
    Spaceline::compress_x_simd_sampling(&mut pixels);
    println!("pixels: {pixels:?}");
    assert_eq!(pixels, reference);
}
