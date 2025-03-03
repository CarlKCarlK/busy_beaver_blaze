use std::fs;

use aligned_vec::AVec;
use busy_beaver_blaze::*;
use thousands::Separable;
use wasm_bindgen_test::wasm_bindgen_test;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[test]
fn bb5_champ() -> Result<(), Error> {
    let mut machine: Machine = BB5_CHAMP.parse()?;

    let debug_interval = 10_000_000;
    let step_count = machine.debug_count(debug_interval);

    println!(
        "Final: Steps {}: {:?}, #1's {}",
        step_count.separate_with_commas(),
        machine,
        machine.count_ones()
    );

    assert_eq!(step_count, 47_176_870);
    assert_eq!(machine.count_ones(), 4098);
    assert_eq!(machine.state(), 7);
    assert_eq!(machine.tape_index(), -12242);

    Ok(())
}

#[wasm_bindgen_test]
#[test]
fn bb5_champ_js() -> Result<(), String> {
    let mut machine: Machine = Machine::from_string(BB5_CHAMP)?;

    let early_stop_some = true;
    let early_stop_number = 50_000_000;
    let step_count = machine.count_js(early_stop_some, early_stop_number);

    println!(
        "Final: Steps {}: {:?}, #1's {}",
        step_count.separate_with_commas(),
        machine,
        machine.count_ones()
    );

    assert_eq!(step_count, 47_176_870);
    assert_eq!(machine.count_ones(), 4098);
    assert_eq!(machine.state(), 7);
    assert_eq!(machine.tape_index(), -12242);

    Ok(())
}

#[wasm_bindgen_test]
#[test]
fn bb5_champ_space_time_js() -> Result<(), String> {
    let program_string = BB5_CHAMP;
    let goal_x: u32 = 1000;
    let goal_y: u32 = 1000;
    let x_smoothness: PowerOfTwo = PowerOfTwo::ONE;
    let y_smoothness: PowerOfTwo = PowerOfTwo::ONE;
    let buffer1_count: PowerOfTwo = PowerOfTwo::ONE;
    let n = 1_000_000;
    let mut space_time_machine = SpaceTimeMachine::from_str(
        program_string,
        goal_x,
        goal_y,
        x_smoothness.log2(),
        y_smoothness.log2(),
        buffer1_count.log2(),
    )?;

    while space_time_machine.nth_js(n - 1) {
        println!(
            "Index {}: {:?}, #1's {}",
            space_time_machine.step_index().separate_with_commas(),
            space_time_machine.machine(),
            space_time_machine.count_ones()
        );
    }

    println!(
        "Final: Steps {}: {:?}, #1's {}",
        space_time_machine.step_index().separate_with_commas(),
        space_time_machine.machine(),
        space_time_machine.count_ones()
    );

    let png_data = space_time_machine.png_data();
    fs::write("tests/expected/test_js.png", &png_data).map_err(|error| error.to_string())?;

    assert_eq!(space_time_machine.step_index() + 1, 47_176_870);
    assert_eq!(space_time_machine.count_ones(), 4098);
    assert_eq!(space_time_machine.state(), 7);
    assert_eq!(space_time_machine.tape_index(), -12242);

    Ok(())
}

#[wasm_bindgen_test]
#[test]
fn seconds_bb5_champ_space_time_js() -> Result<(), String> {
    let program_string = BB5_CHAMP;
    let goal_x: u32 = 1000;
    let goal_y: u32 = 1000;
    let x_smoothness: PowerOfTwo = PowerOfTwo::ONE;
    let y_smoothness: PowerOfTwo = PowerOfTwo::ONE;
    let buffer1_count: PowerOfTwo = PowerOfTwo::ONE; // cmk000000
    let seconds = 0.25;
    let mut space_time_machine = SpaceTimeMachine::from_str(
        program_string,
        goal_x,
        goal_y,
        x_smoothness.log2(),
        y_smoothness.log2(),
        buffer1_count.log2(),
    )?;

    while space_time_machine.step_for_secs_js(seconds, None, 100_000) {
        println!(
            "Index {}: {:?}, #1's {}",
            space_time_machine.step_index().separate_with_commas(),
            space_time_machine.machine(),
            space_time_machine.count_ones()
        );
    }

    println!(
        "Final: Steps {}: {:?}, #1's {}",
        space_time_machine.step_index().separate_with_commas(),
        space_time_machine.machine(),
        space_time_machine.count_ones()
    );

    let png_data = space_time_machine.png_data();
    fs::write("tests/expected/test2_js.png", &png_data)
        .map_err(|error: std::io::Error| error.to_string())?;

    assert_eq!(space_time_machine.step_index() + 1, 47_176_870);
    assert_eq!(space_time_machine.count_ones(), 4098);
    assert_eq!(space_time_machine.state(), 7);
    assert_eq!(space_time_machine.tape_index(), -12242);

    Ok(())
}

#[wasm_bindgen_test]
#[test]
fn machine_7_135_505() -> Result<(), Error> {
    let _machine_a: Machine = MACHINE_7_135_505_A.parse()?;
    let _machine_b: Machine = MACHINE_7_135_505_B.parse()?;
    Ok(())
}

// Create a test that runs bb5 champ to halting and then prints the time it took
// to run the test
// cmk which of these should be bindgen tests?
#[wasm_bindgen_test]
#[test]
fn bb5_champ_time() {
    let start = std::time::Instant::now();
    let step_count = 1 + BB5_CHAMP.parse::<Machine>().unwrap().count();
    let duration = start.elapsed();
    println!(
        "Steps: {}, Duration: {:?}",
        step_count.separate_with_commas(),
        duration
    );
    assert_eq!(step_count, 47_176_870);
}

#[test]
fn benchmark1() -> Result<(), String> {
    let start = std::time::Instant::now();
    let program_string = BB6_CONTENDER;
    let goal_x: u32 = 360;
    let goal_y: u32 = 432;
    let x_smoothness: PowerOfTwo = PowerOfTwo::ONE;
    let y_smoothness: PowerOfTwo = PowerOfTwo::ONE;
    let buffer1_count: PowerOfTwo = PowerOfTwo::ONE; // cmk0000000
    let n = 500_000_000;
    let mut space_time_machine = SpaceTimeMachine::from_str(
        program_string,
        goal_x,
        goal_y,
        x_smoothness.log2(),
        y_smoothness.log2(),
        buffer1_count.log2(),
    )?;

    space_time_machine.nth_js(n - 1);

    println!("Elapsed: {:?}", start.elapsed());

    println!(
        "Final: Steps {}: {:?}, #1's {}",
        space_time_machine.step_index().separate_with_commas(),
        space_time_machine.machine(),
        space_time_machine.count_ones()
    );

    assert_eq!(space_time_machine.step_index(), n);
    assert_eq!(space_time_machine.count_ones(), 10669);
    assert_eq!(space_time_machine.state(), 1);
    assert_eq!(space_time_machine.tape_index(), 34054);

    // cmk LATER what is one method png_data and another to to_png?
    let start2 = std::time::Instant::now();
    let png_data = space_time_machine.png_data();
    fs::write("tests/expected/bench.png", &png_data).unwrap(); // cmk handle error
    println!("Elapsed png: {:?}", start2.elapsed());
    Ok(())
}

#[allow(clippy::shadow_reuse)]
#[test]
#[wasm_bindgen_test]
fn benchmark2() -> Result<(), String> {
    // let start = std::time::Instant::now();
    let early_stop = Some(1_000_000_000);

    let program_string = BB6_CONTENDER;
    let goal_x: u32 = 360;
    let goal_y: u32 = 432;
    let x_smoothness: PowerOfTwo = PowerOfTwo::from_exp(0);
    let y_smoothness: PowerOfTwo = PowerOfTwo::from_exp(0);
    let buffer1_count: PowerOfTwo = PowerOfTwo::ONE; // cmk0000000
    let mut space_time_machine = SpaceTimeMachine::from_str(
        program_string,
        goal_x,
        goal_y,
        x_smoothness.log2(),
        y_smoothness.log2(),
        buffer1_count.log2(),
    )?;

    let chunk_size = 100_000_000;
    let mut total_steps = 1; // Start at 1 since first step is already taken

    loop {
        if early_stop.is_some_and(|early_stop| total_steps >= early_stop) {
            break;
        }

        // Calculate next chunk size
        let next_chunk = if total_steps == 1 {
            chunk_size - 1
        } else {
            chunk_size
        };

        let next_chunk = early_stop.map_or(next_chunk, |early_stop| {
            let remaining = early_stop - total_steps;
            remaining.min(next_chunk)
        });

        // Run the next chunk
        let continues = space_time_machine.nth_js(next_chunk - 1);
        total_steps += next_chunk;

        // Send intermediate update
        println!(
            "intermediate: {:?} Steps {}: {:?}, #1's {}",
            0,
            // start.elapsed(),
            space_time_machine.step_index().separate_with_commas(),
            space_time_machine.machine(),
            space_time_machine.count_ones()
        );

        // let _png_data = space_time_machine.png_data();

        // Exit if machine halted
        if !continues {
            break;
        }
    }

    // Send final result

    println!(
        "Final: {:?} Steps {}: {:?}, #1's {}",
        0, // start.elapsed(),
        space_time_machine.step_index().separate_with_commas(),
        space_time_machine.machine(),
        space_time_machine.count_ones(),
    );

    // // cmk LATER what is one method png_data and another to to_png?
    // let start = std::time::Instant::now();
    // let png_data = space_time_machine.png_data();
    // fs::write("tests/expected/bench2.png", &png_data).unwrap(); // cmk handle error
    // println!("Elapsed png: {:?}", start.elapsed());
    Ok(())
}

// #[test]
#[allow(dead_code)]
fn benchmark3() -> Result<(), String> {
    println!("Smoothness\tSteps\tOnes\tTime(ms)");

    for smoothness in 0..=63 {
        let start = std::time::Instant::now();
        let program_string = BB5_CHAMP;
        let goal_x: u32 = 360;
        let goal_y: u32 = 432;
        let x_smoothness = PowerOfTwo::from_exp(smoothness);
        let y_smoothness = PowerOfTwo::from_exp(smoothness);
        let buffer1_count = PowerOfTwo::ONE; // cmk0000000

        let mut space_time_machine = SpaceTimeMachine::from_str(
            program_string,
            goal_x,
            goal_y,
            x_smoothness.log2(),
            y_smoothness.log2(),
            buffer1_count.log2(),
        )?;

        // Run to completion
        while space_time_machine.nth_js(1_000_000 - 1) {}

        let elapsed = start.elapsed().as_millis();
        println!(
            "{}\t{}\t{}\t{}",
            smoothness,
            space_time_machine.step_count(),
            space_time_machine.count_ones(),
            elapsed
        );

        // Generate PNG for first and last iteration
        if smoothness == 0 || smoothness == 63 {
            let png_data = space_time_machine.png_data();
            fs::write(
                format!("tests/expected/bench3_smooth{smoothness}.png"),
                &png_data,
            )
            .map_err(|error| error.to_string())?;
        }
    }

    Ok(())
}

#[allow(clippy::shadow_reuse)]
#[test]
#[wasm_bindgen_test]
fn benchmark63() -> Result<(), String> {
    // let start = std::time::Instant::now();

    // let early_stop = Some(10_000_000_000);
    // let chunk_size = 100_000_000;
    let early_stop = Some(50_000_000);
    let chunk_size = 5_000_000;
    // let early_stop = Some(250_000_000);
    // let chunk_size = 25_000_000;
    // let early_stop = Some(5_000_000);
    // let chunk_size = 500_000;
    let goal_x: u32 = 360;
    let goal_y: u32 = 432;
    // let goal_x: u32 = 1920;
    // let goal_y: u32 = 1080;

    let program_string = BB6_CONTENDER;
    let x_smoothness: PowerOfTwo = PowerOfTwo::from_exp(63); // cmk0000 63);
    let y_smoothness: PowerOfTwo = PowerOfTwo::from_exp(63);
    let buffer1_count: PowerOfTwo = PowerOfTwo::from_exp(0);
    let mut space_time_machine = SpaceTimeMachine::from_str(
        program_string,
        goal_x,
        goal_y,
        x_smoothness.log2(),
        y_smoothness.log2(),
        buffer1_count.log2(),
    )?;

    let mut total_steps = 1; // Start at 1 since first step is already taken

    loop {
        if early_stop.is_some_and(|early_stop| total_steps >= early_stop) {
            break;
        }

        // Calculate next chunk size
        let next_chunk = if total_steps == 1 {
            chunk_size - 1
        } else {
            chunk_size
        };

        let next_chunk = early_stop.map_or(next_chunk, |early_stop| {
            let remaining = early_stop - total_steps;
            remaining.min(next_chunk)
        });

        // Run the next chunk
        let continues = space_time_machine.nth_js(next_chunk - 1);
        total_steps += next_chunk;

        // Send intermediate update
        println!(
            "intermediate: {:?} Steps {}: {:?}, #1's {}",
            0,
            // start.elapsed(),
            space_time_machine.step_index().separate_with_commas(),
            space_time_machine.machine(),
            space_time_machine.count_ones()
        );

        // let _png_data = space_time_machine.png_data();

        // Exit if machine halted
        if !continues {
            break;
        }
    }

    // Send final result

    println!(
        "Final: {:?} Steps {}: {:?}, #1's {}",
        0, // start.elapsed(),
        space_time_machine.step_index().separate_with_commas(),
        space_time_machine.machine(),
        space_time_machine.count_ones(),
    );

    // cmk LATER what is one method png_data and another to to_png?
    let start = std::time::Instant::now();
    let png_data = space_time_machine.png_data();
    fs::write("tests/expected/bench63.png", &png_data).unwrap(); // cmk handle error
    println!("Elapsed png: {:?}", start.elapsed());
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

    // Rayon is slower, but is it correct?
    let result = average_with_simd_rayon::<64>(&values, step, 2);
    assert_eq!(result, expected);

    // Is count_ones correct?
    let result = average_with_simd_count_ones64(&values, step);
    assert_eq!(result, expected);

    let result = average_with_simd_push::<64>(&values, step);
    assert_eq!(result, expected);
}
