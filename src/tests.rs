#[cfg(test)]
    use crate::{BB5_CHAMP, Error, Machine, PowerOfTwo, SampledSpaceTime};
    use std::fs;
    use thousands::Separable;
    /// See <https://en.wikipedia.org/wiki/Busy_beaver>
    #[allow(clippy::shadow_reuse, clippy::integer_division_remainder_used)]
    #[test]
    fn bb5_champ_space_time_native() -> Result<(), Error> {
        let mut machine: Machine = BB5_CHAMP.parse()?; // cmk
        // let mut machine: Machine = BB6_CONTENDER.parse()?;

        let goal_x: u32 = 1000;
        let goal_y: u32 = 1000;
        let x_smoothness: PowerOfTwo = PowerOfTwo::ONE;
        let y_smoothness: PowerOfTwo = PowerOfTwo::ONE;
        let buffer1_count: PowerOfTwo = PowerOfTwo::ONE; // cmk000000
        let mut sample_space_time =
            SampledSpaceTime::new(goal_x, goal_y, x_smoothness, y_smoothness, buffer1_count);

        let early_stop = Some(10_500_000);
        // let early_stop = Some(1_000_000);
        let debug_interval = Some(1_000_000);

        while let Some(previous_tape_index) = machine
            .next()
            .filter(|_| early_stop.is_none_or(|stop| sample_space_time.step_index + 1 < stop))
        {
            if debug_interval
                .is_none_or(|debug_interval| sample_space_time.step_index % debug_interval == 0)
            {
                println!(
                    "Step {}: {:?},\t{}",
                    sample_space_time.step_index.separate_with_commas(),
                    machine,
                    machine.tape.index_range_to_string(-10..=10)
                );
            }

            sample_space_time.snapshot(&machine, previous_tape_index);
            // let _ = sample_space_time.to_png();
        }

        let png_data = sample_space_time.to_png()?;
        fs::write("tests/expected/test.png", &png_data).unwrap(); // cmk handle error

        println!(
            "Final: Steps {}: {:?}, #1's {}",
            sample_space_time.step_index.separate_with_commas(),
            machine,
            machine.count_ones()
        );

        if early_stop.is_none() {
            assert_eq!(sample_space_time.step_index, 47_176_870);
            assert_eq!(machine.count_ones(), 4098);
            assert_eq!(machine.state(), 7);
            assert_eq!(machine.tape_index(), -12242);
        }

        Ok(())
    }

