// TODO notes
// This is good. To refine the API and structure further, need to know our use case more, e.g. using it in
// the larger program.
// Other things to do:
// -- support more symbols
// -- use derive macros to make the use of the asm template more ergonomic.
//           we might be able to more optimizations, e.g. if both 0/1 branch write same value or move same dir.
// -- Do JIT compilation of the asm template at runtime, to support arbitrary TMs.For example with `dynasmrt` crate.`
//            (see private https://chatgpt.com/c/68c34ed8-b580-8323-99de-56334f3b227b discussion.)

#![deny(clippy::pedantic)]
use clap::Parser;
use busy_beaver_blaze::{CompiledFnId, Config, ConfigError};

#[derive(Debug, Parser, Clone)]
#[command(
    name = "bb5_champ_fast",
    about = "Fast Turing machine runner with inline asm"
)]
struct Args {
    #[arg(long, value_enum, default_value_t = CompiledFnId::Bb5Champ)]
    program: CompiledFnId,

    /// Status print interval in steps
    #[arg(long, value_parser = parse_clean::<u64>, default_value_t = 10_000_000)]
    interval: u64,

    /// Stop after this many steps if provided
    #[arg(long, value_parser = parse_clean::<u64>, default_value_t = u64::MAX)]
    max_steps: u64,

    /// Minimum tape length (includes two sentinel cells)
    #[arg(long, default_value = "2_097_152", value_parser = parse_clean::<usize>)]
    min_tape: usize,

    /// Maximum allowed total tape length (cells incl. sentinels)
    #[arg(long, default_value = "16_777_216", value_parser = parse_clean::<usize>)]
    max_tape: usize,

    /// Suppress progress output
    #[arg(long, default_value_t = false)]
    quiet: bool,
}

impl TryFrom<Args> for Config {
    type Error = ConfigError;

    fn try_from(
        Args {
            program,
            interval,
            max_steps,
            min_tape,
            max_tape,
            quiet,
        }: Args,
    ) -> Result<Self, Self::Error> {
        Self::new(program, interval, max_steps, min_tape, max_tape)
            .map(|config| config.with_quiet(quiet))
    }
}

fn parse_clean<T>(s: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    s.replace(['_', ','], "")
        .parse::<T>()
        .map_err(|e| e.to_string())
}

fn main() {
    let config: Config = Args::parse().try_into().unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let _summary = config.run();
}

#[cfg(test)]
mod tests {
    use super::*;
    use busy_beaver_blaze::{Machine, RunTermination};

    #[test]
    fn args_try_from_copies_fields() {
        let args = Args {
            program: CompiledFnId::Bb6Contender,
            interval: 42,
            max_steps: 7,
            min_tape: 128,
            max_tape: 256,
            quiet: false,
        };
        let config: Config = args.try_into().expect("conversion should succeed");
        assert_eq!(config.interval.get(), 42);
        assert_eq!(config.max_steps, 7);
        assert_eq!(config.min_tape, 128);
        assert_eq!(config.max_tape, 256);
    }

    #[test]
    fn args_try_from_rejects_zero_interval() {
        let args = Args {
            program: CompiledFnId::Bb5Champ,
            interval: 0,
            max_steps: 1,
            min_tape: 128,
            max_tape: 256,
            quiet: false,
        };
        match Config::try_from(args) {
            Err(ConfigError::IntervalTooSmall { interval }) => assert_eq!(interval, 0),
            other => panic!("expected IntervalTooSmall error, got {:?}", other),
        }
    }

    #[test]
    fn args_try_from_rejects_short_tape() {
        let args = Args {
            program: CompiledFnId::Bb5Champ,
            interval: 1_000,
            max_steps: 1,
            min_tape: 2,
            max_tape: 64,
            quiet: false,
        };
        match Config::try_from(args) {
            Err(ConfigError::TapeTooShort { min_tape }) => assert_eq!(min_tape, 2),
            other => panic!("expected TapeTooShort error, got {:?}", other),
        }
    }

    #[test]
    fn args_try_from_rejects_small_max_tape() {
        let args = Args {
            program: CompiledFnId::Bb5Champ,
            interval: 1_000,
            max_steps: 1,
            min_tape: 4,
            max_tape: 2,
            quiet: false,
        };
        match Config::try_from(args) {
            Err(ConfigError::MaxTapeTooSmall { max_tape }) => assert_eq!(max_tape, 2),
            other => panic!("expected MaxTapeTooSmall error, got {:?}", other),
        }
    }

    #[test]
    fn run_stops_at_max_steps() {
        let compiled_machine: Config = Args {
            program: CompiledFnId::Bb5Champ,
            interval: 1_000_000,
            max_steps: 1_000,
            min_tape: 128,
            max_tape: 1usize << 16,
            quiet: true,
        }
        .try_into()
        .expect("conversion should succeed");

        let summary = compiled_machine.run();
        assert_eq!(summary.step_count, 1_000);
        assert_eq!(summary.run_termination, RunTermination::MaxSteps);
        assert!(summary.state_index <= 4);
        assert!(summary.elapsed_secs >= 0.0);
        assert!(summary.tape().len() >= 1);
    }

    #[test]
    fn bb5_champ_halts_at_47_million_steps() {
        let compiled_machine: Config = Args {
            program: CompiledFnId::Bb5Champ,
            interval: 20_000_000,
            max_steps: u64::MAX,
            min_tape: 2_097_152,
            max_tape: 16_777_216,
            quiet: true,
        }
        .try_into()
        .expect("conversion should succeed");

        let summary = compiled_machine.run();
        assert_eq!(summary.run_termination, RunTermination::Halted);
        assert_eq!(summary.step_count, 47_176_870);
    }

    #[test]
    fn bb5_compiled_does_not_halt_early() {
        let compiled_machine: Config = Args {
            program: CompiledFnId::Bb5Champ,
            interval: 10_000_000,
            max_steps: 20_000_000,
            min_tape: 2_097_152,
            max_tape: 16_777_216,
            quiet: true,
        }
        .try_into()
        .expect("conversion should succeed");

        let summary = compiled_machine.run();
        assert_eq!(summary.run_termination, RunTermination::MaxSteps);
        assert_eq!(summary.step_count, 20_000_000);
        assert!(summary.state_index <= 4);
    }

    #[test]
    fn bb5_run() -> Result<(), ConfigError> {
        let summary = Config::new(
            CompiledFnId::Bb5Champ,
            10_000_000,
            u64::MAX,
            2_097_152,
            16_777_216,
        )?
        .with_quiet(true)
        .run();

        assert_eq!(summary.run_termination, RunTermination::Halted);
        assert_eq!(summary.step_count, 47_176_870);
        assert_eq!(summary.state_index, 4);
        Ok(())
    }

    #[test]
    fn bb33_compiled_halts_and_has_expected_nonzeros() -> Result<(), ConfigError> {
        // BB(3,3): 1RB 2LA 1RA_1LA 1RZ 1RC_2RB 1RC 2RB
        // Expect: halts at exactly 355,317 steps and has 772 nonzeros.
        let summary = Config::new(
            CompiledFnId::Bb33_355K,
            10_000_000,
            u64::MAX,
            2_097_152,
            16_777_216,
        )?
        .with_quiet(true)
        .run();

        assert_eq!(summary.run_termination, RunTermination::Halted);
        assert_eq!(summary.step_count, 355_317);
        // Cross-check with interpreter to avoid ambiguity in symbol tallies.
        let mut interpreter =
            Machine::from_string("1RB2LA1RA_1LA1RZ1RC_2RB1RC2RB").expect("parse BB(3,3)");
        let mut steps = 1u64;
        while interpreter.step() {
            steps += 1;
        }
        assert_eq!(steps, summary.step_count);
        let nonblanks = interpreter.count_nonblanks() as usize;
        let compiled_nonblanks = summary.tape().iter().filter(|&&v| v != 0).count();
        assert_eq!(compiled_nonblanks, nonblanks);
        assert_eq!(nonblanks, 772);
        Ok(())
    }
}
