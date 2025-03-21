use std::fs;

use busy_beaver_blaze::{
    BB5_CHAMP, BB6_CONTENDER, DebuggableIterator, Error, MACHINE_7_135_505_A, MACHINE_7_135_505_B,
    Machine, SpaceByTimeMachine,
};
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
fn bb5_champ_space_by_time_js() -> Result<(), String> {
    let program_string = BB5_CHAMP;
    let goal_x: u32 = 1000;
    let goal_y: u32 = 1000;
    let binning = false;
    let n = 1_000_000;
    let mut space_by_time_machine =
        SpaceByTimeMachine::from_str(program_string, goal_x, goal_y, binning, 0)?;

    while space_by_time_machine.nth_js(n - 1) {
        println!(
            "Index {}: {:?}, #1's {}",
            space_by_time_machine.step_index().separate_with_commas(),
            space_by_time_machine.machine(),
            space_by_time_machine.count_ones()
        );
    }

    println!(
        "Final: Steps {}: {:?}, #1's {}",
        space_by_time_machine.step_index().separate_with_commas(),
        space_by_time_machine.machine(),
        space_by_time_machine.count_ones()
    );

    let png_data = space_by_time_machine.png_data();
    fs::write("tests/expected/test_js.png", &png_data).map_err(|error| error.to_string())?;

    assert_eq!(space_by_time_machine.step_index() + 1, 47_176_870);
    assert_eq!(space_by_time_machine.count_ones(), 4098);
    assert_eq!(space_by_time_machine.state(), 7);
    assert_eq!(space_by_time_machine.tape_index(), -12242);

    Ok(())
}

#[wasm_bindgen_test]
#[test]
fn seconds_bb5_champ_space_by_time_js() -> Result<(), String> {
    let program_string = BB5_CHAMP;
    let goal_x: u32 = 1000;
    let goal_y: u32 = 1000;
    let binning = false;
    let seconds = 0.25;
    let mut space_by_time_machine =
        SpaceByTimeMachine::from_str(program_string, goal_x, goal_y, binning, 0)?;

    while space_by_time_machine.step_for_secs_js(seconds, None, 100_000) {
        println!(
            "Index {}: {:?}, #1's {}",
            space_by_time_machine.step_index().separate_with_commas(),
            space_by_time_machine.machine(),
            space_by_time_machine.count_ones()
        );
    }

    println!(
        "Final: Steps {}: {:?}, #1's {}",
        space_by_time_machine.step_index().separate_with_commas(),
        space_by_time_machine.machine(),
        space_by_time_machine.count_ones()
    );

    let png_data = space_by_time_machine.png_data();
    fs::write("tests/expected/test2_js.png", &png_data)
        .map_err(|error: std::io::Error| error.to_string())?;

    assert_eq!(space_by_time_machine.step_index() + 1, 47_176_870);
    assert_eq!(space_by_time_machine.count_ones(), 4098);
    assert_eq!(space_by_time_machine.state(), 7);
    assert_eq!(space_by_time_machine.tape_index(), -12242);

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
    let binning = false;
    let n = 500_000_000;
    let mut space_by_time_machine =
        SpaceByTimeMachine::from_str(program_string, goal_x, goal_y, binning, 0)?;

    space_by_time_machine.nth_js(n - 1);

    println!("Elapsed: {:?}", start.elapsed());

    println!(
        "Final: Steps {}: {:?}, #1's {}",
        space_by_time_machine.step_index().separate_with_commas(),
        space_by_time_machine.machine(),
        space_by_time_machine.count_ones()
    );

    assert_eq!(space_by_time_machine.step_index(), n);
    assert_eq!(space_by_time_machine.count_ones(), 10669);
    assert_eq!(space_by_time_machine.state(), 1);
    assert_eq!(space_by_time_machine.tape_index(), 34054);

    // cmk LATER what is one method png_data and another to to_png?
    let start2 = std::time::Instant::now();
    let png_data = space_by_time_machine.png_data();
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
    let binning = false;
    let mut space_by_time_machine =
        SpaceByTimeMachine::from_str(program_string, goal_x, goal_y, binning, 0)?;

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
        let continues = space_by_time_machine.nth_js(next_chunk - 1);
        total_steps += next_chunk;

        // Send intermediate update
        println!(
            "intermediate: {:?} Steps {}: {:?}, #1's {}",
            0,
            // start.elapsed(),
            space_by_time_machine.step_index().separate_with_commas(),
            space_by_time_machine.machine(),
            space_by_time_machine.count_ones()
        );

        // let _png_data = space_by_time_machine.png_data();

        // Exit if machine halted
        if !continues {
            break;
        }
    }

    // Send final result

    println!(
        "Final: {:?} Steps {}: {:?}, #1's {}",
        0, // start.elapsed(),
        space_by_time_machine.step_index().separate_with_commas(),
        space_by_time_machine.machine(),
        space_by_time_machine.count_ones(),
    );

    // // cmk LATER what is one method png_data and another to to_png?
    // let start = std::time::Instant::now();
    // let png_data = space_by_time_machine.png_data();
    // fs::write("tests/expected/bench2.png", &png_data).unwrap(); // cmk handle error
    // println!("Elapsed png: {:?}", start.elapsed());
    Ok(())
}

// #[test]
#[allow(dead_code)]
fn benchmark3() -> Result<(), String> {
    println!("pixel_policy\tSteps\tOnes\tTime(ms)");

    for binning in [false, true] {
        let start = std::time::Instant::now();
        let program_string = BB5_CHAMP;
        let goal_x: u32 = 360;
        let goal_y: u32 = 432;

        let mut space_by_time_machine =
            SpaceByTimeMachine::from_str(program_string, goal_x, goal_y, binning, 0)?;

        // Run to completion
        while space_by_time_machine.nth_js(1_000_000 - 1) {}

        let elapsed = start.elapsed().as_millis();
        println!(
            "{}\t{}\t{}\t{}",
            binning,
            space_by_time_machine.step_count(),
            space_by_time_machine.count_ones(),
            elapsed
        );

        // Generate PNG for first and last iteration
        let png_data = space_by_time_machine.png_data();
        fs::write(
            format!("tests/expected/bench3_smooth{binning}.png"),
            &png_data,
        )
        .map_err(|error| error.to_string())?;
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
    let binning = false;

    let program_string = BB6_CONTENDER;
    let mut space_by_time_machine =
        SpaceByTimeMachine::from_str(program_string, goal_x, goal_y, binning, 0)?;

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
        let continues = space_by_time_machine.nth_js(next_chunk - 1);
        total_steps += next_chunk;

        // Send intermediate update
        println!(
            "intermediate: {:?} Steps {}: {:?}, #1's {}",
            0,
            // start.elapsed(),
            space_by_time_machine.step_index().separate_with_commas(),
            space_by_time_machine.machine(),
            space_by_time_machine.count_ones()
        );

        // let _png_data = space_by_time_machine.png_data();

        // Exit if machine halted
        if !continues {
            break;
        }
    }

    // Send final result

    println!(
        "Final: {:?} Steps {}: {:?}, #1's {}",
        0, // start.elapsed(),
        space_by_time_machine.step_index().separate_with_commas(),
        space_by_time_machine.machine(),
        space_by_time_machine.count_ones(),
    );

    // cmk LATER what is one method png_data and another to to_png?
    let start = std::time::Instant::now();
    let png_data = space_by_time_machine.png_data();
    fs::write("tests/expected/bench63.png", &png_data).unwrap(); // cmk handle error
    println!("Elapsed png: {:?}", start.elapsed());
    Ok(())
}
