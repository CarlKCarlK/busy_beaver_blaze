use crate::{
    ALIGN, BB5_CHAMP, BB6_CONTENDER, Error, Machine, PixelPolicy, PowerOfTwo, SpaceByTime,
    SpaceByTimeMachine, average_with_iterators, average_with_simd, average_with_simd_count_ones64,
    average_with_simd_push, bool_u8::BoolU8, find_stride, pixel::Pixel, spaceline::Spaceline,
    spacelines::Spacelines,
};
use aligned_vec::AVec;
use rand::{Rng, SeedableRng};
use std::{
    collections::{HashMap, HashSet},
    fs,
};
use thousands::Separable;

#[allow(clippy::shadow_reuse, clippy::integer_division_remainder_used)]
#[test]
fn bb5_champ_space_by_time_native() -> Result<(), Error> {
    let mut machine: Machine = BB5_CHAMP.parse()?; // cmk
    // let mut machine: Machine = BB6_CONTENDER.parse()?;

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
    )?;
    fs::write("tests/expected/test.png", &png_data).unwrap(); // cmk handle error

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

    // Is count_ones correct?
    let result = average_with_simd_count_ones64(&values, step);
    assert_eq!(result, expected);

    let result = average_with_simd_push::<64>(&values, step);
    assert_eq!(result, expected);
}

// cmk00000 test on binning true vs false with different numbers of part_counts and check that get the same png bytes.
#[allow(clippy::shadow_reuse, clippy::too_many_lines)]
#[test]
fn parts() {
    let max_rows = 2413u64;
    let part_count = 3;
    let goal_x: u32 = 360;
    let goal_y: u32 = 30;
    let binning = true;

    // let max_rows = 250_000_000u64;
    // let part_count = 16;
    // let goal_x: u32 = 360;
    // let goal_y: u32 = 432;
    // let binning = true;

    // let max_rows = 10_000_000u64;
    // let part_count = 16;
    // let goal_x: u32 = 360;
    // let goal_y: u32 = 30;
    // let binning = true;

    // let max_rows = 1_000_000u64;
    // let part_count = 1;
    // let goal_x: u32 = 360;
    // let goal_y: u32 = 30;
    // let binning = true;

    // let max_rows = 300u64;
    // let part_count = 2;
    // let goal_x: u32 = 360;
    // let goal_y: u32 = 432;
    // let binning = true;

    // let max_rows = 5u64;
    // let part_count = 10;

    // let goal_x: u32 = 360;
    // let goal_y: u32 = 432;
    let program_string = BB6_CONTENDER;
    let mut space_by_time_machine_first = SpaceByTimeMachine::from_str_in_parts(
        max_rows,
        part_count,
        program_string,
        goal_x,
        goal_y,
        binning,
    );
    let png_data = space_by_time_machine_first.png_data();
    fs::write("tests/expected/part.png", &png_data).unwrap(); // cmk handle error

    // assert!(len < goal_y as usize * 2, "real assert 2");
}

#[test]
fn resample_simd() {
    for len in [0, 1, 2, 3, 5, 101, 111, 4001] {
        let seed = 0;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut pixels: AVec<Pixel, _> =
            AVec::from_iter(ALIGN, (0..len).map(|_| rng.random::<bool>().into()));

        // println!("len {len} before: {pixels:?}");
        let mut reference = pixels.clone();
        Spaceline::resample_one(&mut reference);
        // println!("reference: {reference:?}");
        Spaceline::resample_one(&mut pixels);
        // println!("after: {pixels:?}");
        assert_eq!(pixels, reference);
    }
}

#[test]
fn test_find_stride() {
    for tape_neg_len in [0usize, 1, 2, 3, 4, 5, 10, 101, 9999, 1_000_007] {
        for tape_non_neg_len in [1usize, 2, 3, 4, 5, 10, 101, 9999, 1_000_007] {
            let tape_len = tape_neg_len + tape_non_neg_len;
            for goal_x in [2, 3, 4, 5, 6, 7, 10, 100, 1000, 33_333] {
                let x_stride = find_stride(tape_neg_len, tape_non_neg_len, goal_x);
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
fn combo_parts() {
    // make a HashMap from the strings "BB5_CHAMP", "BB6_CONTENDER" to the string constants
    let program_name_to_string =
        HashMap::from([("BB5_CHAMP", BB5_CHAMP), ("BB6_CONTENDER", BB6_CONTENDER)]);

    let exception_set = HashSet::from(["ignore"]);

    for early_stop in [2, 5u64, 6, 7, 300u64, 1_000_000u64] {
        for goal_x in [2, 30, 360] {
            for goal_y in [2, 3, 4, 30, 432] {
                for program_name in ["BB5_CHAMP", "BB6_CONTENDER"] {
                    for binning in [false, true] {
                        // cmk0000000 for binning in [true, false] {
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
                        println!(
                            "reference_machine: {:?} {:?}",
                            reference_machine.space_by_time.y_stride,
                            reference_machine.space_by_time.spacelines
                        );
                        let (reference_png_data, ref_x, ref_y, reference_packed_data) =
                            reference_machine.png_data_and_packed_data();
                        println!("---------------");
                        for part_count in [1, 2, 5, 16] {
                            let key = format!(
                                "early_stop: {early_stop}, goal_x: {goal_x}, goal_y: {goal_y}, program_name: {program_name}, binning: {binning}, part_count: {part_count}"
                            );
                            println!("{key}");

                            let mut machine = SpaceByTimeMachine::from_str_in_parts(
                                early_stop,
                                part_count,
                                program_string,
                                goal_x,
                                goal_y,
                                binning,
                            );
                            let (png_data, x, y, packed_data) = machine.png_data_and_packed_data();

                            // Must be the same length and a given value can vary by no more than y_stride.log2() + x_stride.log2()
                            let last_spacetime =
                                machine.space_by_time.spacelines.main.last().unwrap();
                            let max_diff = machine.space_by_time.y_stride.log2()
                                + last_spacetime.x_stride.log2();
                            // println!("max_diff: {max_diff}");

                            let mut ok = true;
                            ok = ok && packed_data.len() == reference_packed_data.len();
                            for (ref_val, val) in
                                reference_packed_data.iter().zip(packed_data.iter())
                            {
                                let abs_diff = ref_val.abs_diff(*val);
                                if abs_diff > 0 {
                                    println!("|{ref_val}-{val}|= {abs_diff} ?<= {max_diff}");
                                }
                                ok = ok && ref_val.abs_diff(*val) <= max_diff;
                            }

                            if !ok {
                                if exception_set.contains(key.as_str()) {
                                    println!("Skip: PNG data does not match for \"{key}\"");
                                    continue;
                                }
                                println!(
                                    "goal_x {goal_x}, goal_y {goal_y}, ref_x, {ref_x}, ref_y: {ref_y}, x, {x}, y: {y}"
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
fn one_parts() {
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
    println!(
        "reference_machine: {:?} {:?}",
        reference_machine.space_by_time.y_stride, reference_machine.space_by_time.spacelines
    );
    let (reference_png_data, ref_x, ref_y, ref_packed) =
        reference_machine.png_data_and_packed_data();
    println!("---------------");

    let key = format!(
        "early_stop: {early_stop}, goal_x: {goal_x}, goal_y: {goal_y}, program_name: {program_name}, binning: {binning}, part_count: {part_count}"
    );
    println!("{key}");

    let mut machine = SpaceByTimeMachine::from_str_in_parts(
        early_stop,
        part_count,
        program_string,
        goal_x,
        goal_y,
        binning,
    );
    let (png_data, x, y, packed_data) = machine.png_data_and_packed_data();

    println!("goal_x {goal_x}, goal_y {goal_y}, ref_x, {ref_x}, ref_y: {ref_y}, x, {x}, y: {y}");
    println!("ref_packed: {ref_packed:?}");
    println!("packed_data: {packed_data:?}");
    let ref_file = "tests/expected/one_parts_ref.png";
    fs::write(ref_file, &reference_png_data).unwrap();
    let test_file = "tests/expected/one_parts_test.png";
    fs::write(test_file, &png_data).unwrap();
    assert!(
        (png_data == reference_png_data),
        "PNG data does not match for {key}"
    );
}
