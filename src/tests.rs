use crate::{
    ALIGN, BB5_CHAMP, BB6_CONTENDER, Error, Machine, PixelPolicy, PowerOfTwo, SpaceByTime,
    SpaceByTimeMachine, average_with_iterators, average_with_simd, average_with_simd_count_ones64,
    average_with_simd_push, bool_u8::BoolU8, pixel::Pixel, sample_rate,
};
use aligned_vec::AVec;
use rayon::prelude::*;
use std::fs;
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

    let png_data = sample_space_by_time.to_png()?;
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

#[allow(clippy::shadow_reuse)]
#[test]
fn parts() {
    let max_rows = 250_000_000u64;
    let part_count = 16;
    let goal_x: u32 = 360;
    let goal_y: u32 = 432;
    // // let max_rows = 10_000_000u64;
    // // let part_count = 16;
    // let max_rows = 1_000_000u64;
    // let part_count = 1;

    // let max_rows = 300u64;
    // let part_count = 2;
    // let goal_x: u32 = 360;
    // let goal_y: u32 = 432;

    // let max_rows = 5u64;
    // let part_count = 10;

    // let goal_x: u32 = 360;
    // let goal_y: u32 = 432;
    let program_string = BB6_CONTENDER;
    let binning = true;

    assert!(max_rows > 0); // panic if early_stop is 0
    assert!(part_count > 0); // panic if part_count is 0
    let mut rows_per_part = max_rows.div_ceil(part_count);

    let y_stride = sample_rate(rows_per_part, goal_y);
    rows_per_part += y_stride.offset_to_align(rows_per_part as usize) as u64;
    // assert_eq!(y_stride.double(), sample_rate(rows_per_part, goal_y), "+1?");
    assert!(y_stride.divides_u64(rows_per_part), "even?");

    println!("Part max_rows_per_part: {rows_per_part}");
    let range_list: Vec<_> = (0..max_rows)
        .step_by(rows_per_part as usize)
        .map(|start| start..(start + rows_per_part).min(max_rows))
        .collect();

    let results: Vec<(bool, SpaceByTimeMachine)> = range_list
        .par_iter()
        // .iter() // cmk000000000000
        .enumerate()
        .map(|(part_index, range)| {
            let (start, end) = (range.start, range.end);
            println!("{part_index}: Start: {start}, End: {end}");
            let mut space_by_time_machine =
                SpaceByTimeMachine::from_str(program_string, goal_x, goal_y, binning, start)
                    .expect("Failed to create machine");
            // println!(
            //     "{}: Initial: {:?}",
            //     part_index,
            //     space_by_time_machine.step_index()
            // );
            for _time in start + 1..end {
                // println!("{part_index}: Time: {time}");
                if space_by_time_machine.next().is_none() {
                    break;
                }
            }
            // println!(
            //     "{}: after: {:?}",
            //     part_index,
            //     space_by_time_machine.step_index()
            // );
            (true, space_by_time_machine)
        })
        .collect();

    let mut results_iter = results.into_iter();
    let (_continues0, mut space_by_time_machine0) = results_iter.next().unwrap();
    let space_by_time0 = &mut space_by_time_machine0.space_by_time;
    for (_continues, space_by_time_machine) in results_iter {
        // cmk do something with continues
        let spacelines = space_by_time_machine.space_by_time.spacelines;
        let main = spacelines.main;
        let buffer0 = spacelines.buffer0;
        println!("len buffer0: {}", buffer0.len());
        for spaceline in main {
            let weight = spaceline.stride;
            space_by_time0.push_can_weigh_more(spaceline, weight);
        }
        for (spaceline, weight) in buffer0 {
            space_by_time0.push_can_weigh_more(spaceline, weight);
        }
    }
    // println!(
    //     "Final: {:?} Steps {}: {:?}, #1's {}",
    //     0, // start.elapsed(),
    //     space_by_time_machine_last
    //         .step_index()
    //         .separate_with_commas(),
    //     space_by_time_machine_last.machine(),
    //     space_by_time_machine_last.count_ones(),
    // );
    let png_data = space_by_time_machine0.png_data();
    fs::write(format!("tests/expected/part.png"), &png_data).unwrap(); // cmk handle error
}
