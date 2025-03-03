use crate::{
    ALIGN, average_with_iterators, average_with_simd, average_with_simd_count_ones64,
    average_with_simd_push, bool_u8::BoolU8, pixel::Pixel,
};
#[cfg(test)]
use crate::{BB5_CHAMP, Error, Machine, PowerOfTwo, SpaceByTime};
use aligned_vec::AVec;
use std::fs;
use thousands::Separable;
/// See <https://en.wikipedia.org/wiki/Busy_beaver>
#[allow(clippy::shadow_reuse, clippy::integer_division_remainder_used)]
#[test]
fn bb5_champ_space_by_time_native() -> Result<(), Error> {
    let mut machine: Machine = BB5_CHAMP.parse()?; // cmk
    // let mut machine: Machine = BB6_CONTENDER.parse()?;

    let goal_x: u32 = 1000;
    let goal_y: u32 = 1000;
    let x_smoothness: PowerOfTwo = PowerOfTwo::ONE;
    let y_smoothness: PowerOfTwo = PowerOfTwo::ONE;
    let mut sample_space_by_time = SpaceByTime::new(goal_x, goal_y, x_smoothness, y_smoothness);

    let early_stop = Some(10_500_000);
    // let early_stop = Some(1_000_000);
    let debug_interval = Some(1_000_000);

    while let Some(previous_tape_index) = machine
        .next()
        .filter(|_| early_stop.is_none_or(|stop| sample_space_by_time.step_index + 1 < stop))
    {
        if debug_interval
            .is_none_or(|debug_interval| sample_space_by_time.step_index % debug_interval == 0)
        {
            println!(
                "Step {}: {:?},\t{}",
                sample_space_by_time.step_index.separate_with_commas(),
                machine,
                machine.tape.index_range_to_string(-10..=10)
            );
        }

        sample_space_by_time.snapshot(&machine, previous_tape_index);
        // let _ = sample_space_by_time.to_png();
    }

    let png_data = sample_space_by_time.to_png()?;
    fs::write("tests/expected/test.png", &png_data).unwrap(); // cmk handle error

    println!(
        "Final: Steps {}: {:?}, #1's {}",
        sample_space_by_time.step_index.separate_with_commas(),
        machine,
        machine.count_ones()
    );

    if early_stop.is_none() {
        assert_eq!(sample_space_by_time.step_index, 47_176_870);
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
