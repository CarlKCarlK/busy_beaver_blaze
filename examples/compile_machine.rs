#![allow(named_asm_labels)] // Allow alphabetic labels in inline asm for readability
use clap::{Parser, ValueEnum};
use derive_more::{Display, Error};
use std::{
    num::NonZeroU64,
    time::{Duration, Instant},
};
use thousands::Separable;

// Macro helpers to generate the full asm template from a TM spec
macro_rules! tm_move {
    (R) => {
        /* move head Right */
        "inc rsi\n"
    };
    (L) => {
        /* move head Left */
        "dec rsi\n"
    };
}
macro_rules! tm_next {
    ( $P:ident, HALT, $id:expr ) => {
        concat!(
            /* set status = halt */ "mov al, 1\n",
            /* record current state id */ "mov bl, ",
            stringify!($id),
            "\n",
            /* jump to end label */ "jmp ",
            stringify!($P),
            "_END\n"
        )
    };
    ( $P:ident, $N:ident, $id:expr ) => {
        concat!(
            /* goto next state */ "jmp ",
            stringify!($P),
            "_",
            stringify!($N),
            "\n"
        )
    };
}
macro_rules! tm_dispatch {
    ( $P:ident, $S:ident, $id:expr ) => {
        concat!(
            /* compare BL with state id */ "cmp bl, ",
            stringify!($id),
            "\n",
            /* if equal, jump to state */ "je ",
            stringify!($P),
            "_",
            stringify!($S),
            "\n"
        )
    };
}
macro_rules! tm_state_block {
    ( $P:ident, $S:ident, $id:expr, ( $w0:literal, $d0:ident, $n0:ident ), ( $w1:literal, $d1:ident, $n1:ident ) ) => {
        concat!(
            /* state label */ stringify!($P),
            "_",
            stringify!($S),
            ":\n",
            /* any credit left? */ "cmp rcx, 0\n",
            /* continue if yes */ "jne ",
            stringify!($P),
            "_",
            stringify!($S),
            "_CONT\n",
            /* record resume state */ "mov bl, ",
            stringify!($id),
            "\n",
            /* exit chunk */ "jmp ",
            stringify!($P),
            "_END\n",
            /* continue label */ stringify!($P),
            "_",
            stringify!($S),
            "_CONT:\n",
            /* load cell */ "mov dl, [rsi]\n",
            /* boundary sentinel? */ "cmp dl, 2\n",
            /* jump if boundary */ "je ",
            stringify!($P),
            "_BOUNDARY_",
            stringify!($S),
            "\n",
            /* is cell == 0? */ "test dl, dl\n",
            /* branch if 1 */ "jnz ",
            stringify!($P),
            "_",
            stringify!($S),
            "_ONE\n",
            /* write on 0-branch */ "mov byte ptr [rsi], ",
            stringify!($w0),
            "\n",
            tm_move!($d0),
            /* consume one step */ "sub rcx, 1\n",
            /* jump to next state (0-branch) */ tm_next!($P, $n0, $id),
            /* 1-branch label */ stringify!($P),
            "_",
            stringify!($S),
            "_ONE:\n",
            /* write on 1-branch */ "mov byte ptr [rsi], ",
            stringify!($w1),
            "\n",
            tm_move!($d1),
            /* consume one step */ "sub rcx, 1\n",
            /* jump to next state (1-branch) */ tm_next!($P, $n1, $id),
            /* boundary label */ stringify!($P),
            "_BOUNDARY_",
            stringify!($S),
            ":\n",
            /* record state id */ "mov bl, ",
            stringify!($id),
            "\n",
            /* go to common boundary */ "jmp ",
            stringify!($P),
            "_BOUNDARY\n",
        )
    };
}
macro_rules! tm_prog {
    ( $P:ident, ($S0:ident, $id0:expr, $z0:tt, $o0:tt) $(, ($S:ident, $id:expr, $z:tt, $o:tt) )* $(,)? ) => {
        concat!(
            /* clear status (AL) */ "xor eax, eax\n",
            /* set heartbeat (credit) in RCX */ "mov rcx, {hb}\n",
            tm_dispatch!($P, $S0, $id0),
            $( tm_dispatch!($P, $S, $id), )*
            /* jump to first state */ "jmp ", stringify!($P), "_", stringify!($S0), "\n",
            tm_state_block!($P, $S0, $id0, $z0, $o0),
            $( tm_state_block!($P, $S, $id, $z, $o), )*
            /* common boundary label */ stringify!($P), "_BOUNDARY:\n",
            /* set status = boundary */ "mov al, 2\n",
            /* chunk end */ stringify!($P), "_END:\n",
            /* copy heartbeat to r8 (steps taken temp) */ "mov r8, {hb}\n",
            /* steps_taken = hb - rcx */ "sub r8, rcx\n",
        )
    };
}

// Note: we intentionally accept only plain numeric values for CLI args now.

// Heartbeat is chosen per-iteration; no fixed default needed.
fn format_duration(total_secs: f64) -> String {
    if !total_secs.is_finite() {
        return String::from("--:--:--");
    }
    let clamped = total_secs.max(0.0);
    let duration: Duration = Duration::from_secs_f64(clamped);
    humantime::format_duration(duration).to_string()
}

#[derive(Debug, Parser, Clone)]
#[command(
    name = "bb5_champ_fast",
    about = "Fast Turing machine runner with inline asm"
)]
struct Args {
    #[arg(long, value_enum, default_value_t = ProgramSelect::Bb5Champ)]
    program: ProgramSelect,

    /// Status print interval in steps. Omit to disable periodic reports.
    #[arg(long, value_parser = parse_clean::<u64>, default_value_t = u64::MAX)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum ProgramSelect {
    Bb5Champ,
    Bb6Contender,
}

#[derive(Debug, Clone)]
pub struct CompiledMachine {
    pub min_tape: usize,
    pub interval: NonZeroU64,
    pub max_tape: usize,
    pub max_steps: u64,
    pub program: ProgramSelect,
}

impl TryFrom<Args> for CompiledMachine {
    type Error = Error;

    fn try_from(
        Args {
            program,
            interval,
            max_steps,
            min_tape,
            max_tape,
        }: Args,
    ) -> Result<Self, Self::Error> {
        if min_tape < 3 {
            return Err(Error::TapeTooShort { min_tape });
        }
        if max_tape < min_tape {
            return Err(Error::MaxTapeTooSmall { max_tape });
        }
        let interval = NonZeroU64::new(interval).ok_or(Error::IntervalTooSmall { interval })?;

        Ok(Self {
            min_tape,
            interval,
            max_tape,
            max_steps,
            program,
        })
    }
}

#[derive(Debug, Display, Error)]
pub enum Error {
    #[display("min_tape must be >= 3 (two sentinels + at least one interior); got {min_tape}")]
    TapeTooShort { min_tape: usize },
    #[display("max_tape must be >= 3 (two sentinels + at least one interior); got {max_tape}")]
    MaxTapeTooSmall { max_tape: usize },
    #[display("interval must be > 0; got {interval}")]
    IntervalTooSmall { interval: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunTermination {
    Halted,
    MaxSteps,
    MaxMemoryLeft,
    MaxMemoryRight,
}

#[derive(Debug, Clone, Copy)]
pub struct Summary {
    pub step_count: u64,
    pub final_state_id: u8,
    pub run_termination: RunTermination,
    pub elapsed_seconds: f64,
    pub tape_len: usize,
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
    let config: CompiledMachine = Args::parse().try_into().unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    if let Err(error) = config.run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

impl CompiledMachine {
    pub fn run(self) -> Result<Summary, Error> {
        // cmk don't like that validation is here and not in the constructor
        let CompiledMachine {
            min_tape,
            interval,
            max_tape,
            max_steps,
            program,
        } = self;
        let mut tape_len = min_tape;
        let mut next_report: u64 = interval.get();
        let mut tape: Vec<u8> = vec![0; tape_len];
        tape[0] = 2;
        tape[tape_len - 1] = 2;
        let mut head_pointer = unsafe { tape.as_mut_ptr().add(1 + ((tape_len - 2) >> 1)) };
        let mut step_count: u64 = 0;
        let mut state_id: u8 = 0;
        let start_time: Instant = Instant::now();
        type CompiledFn = unsafe fn(*mut u8, u8, u64) -> (*mut u8, u8, u64, u8);
        let compiled_fn: CompiledFn = match program {
            ProgramSelect::Bb5Champ => bb5_champ_compiled,
            ProgramSelect::Bb6Contender => bb6_contender_compiled,
        };
        loop {
            let hb_to_limit = max_steps.saturating_sub(step_count);
            let hb_to_report = next_report.saturating_sub(step_count);
            let hb_this_chunk: u64 = hb_to_limit.min(hb_to_report).max(1);
            let (new_head, status_code, steps_taken_this_chunk, new_state_id) =
                unsafe { compiled_fn(head_pointer, state_id, hb_this_chunk) };
            head_pointer = new_head;
            step_count += steps_taken_this_chunk;
            state_id = new_state_id;
            if step_count >= max_steps {
                // cmk should this be a result?
                println!(
                    "reached max steps {}; stopping",
                    step_count.separate_with_commas()
                );
                let elapsed_seconds = start_time.elapsed().as_secs_f64();
                println!("{:.3} s", elapsed_seconds);
                return Ok(Summary {
                    step_count,
                    final_state_id: state_id,
                    run_termination: RunTermination::MaxSteps,
                    elapsed_seconds,
                    tape_len,
                });
            }
            if status_code == 0 {
                if step_count >= next_report {
                    let crossed = (step_count - next_report) / interval + 1;
                    let last = next_report + (crossed - 1) * interval.get();
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let done = last as f64; // Convert last to f64 directly
                    let steps_per_sec = if elapsed > 0.0 { done / elapsed } else { 0.0 };
                    let remaining = (max_steps.saturating_sub(last)) as f64;
                    let eta = if steps_per_sec > 0.0 {
                        remaining / steps_per_sec
                    } else {
                        f64::INFINITY
                    };
                    let total = if eta.is_finite() {
                        format_duration(elapsed + eta)
                    } else {
                        String::from("--:--:--")
                    };
                    println!(
                        "{} steps (ETA {:.3} s, total ~ {}, elapsed {:.3} s)",
                        last.separate_with_commas(),
                        total,
                        eta,
                        elapsed
                    );
                    next_report = last.saturating_add(interval.get());
                }
                continue;
            }
            if status_code == 1 {
                println!("halted after {} steps", step_count.separate_with_commas());
                let elapsed_seconds = start_time.elapsed().as_secs_f64();
                println!("{:.3} s", elapsed_seconds);
                return Ok(Summary {
                    step_count,
                    final_state_id: state_id,
                    run_termination: RunTermination::Halted,
                    elapsed_seconds,
                    tape_len,
                });
            }
            let left_sentinel_ptr = tape.as_ptr();
            let right_sentinel_ptr = unsafe { tape.as_ptr().add(tape_len - 1) };
            let hit_left = head_pointer.cast_const() == left_sentinel_ptr;
            let hit_right = head_pointer.cast_const() == right_sentinel_ptr;
            if hit_left {
                match extend_tape_left(&mut tape, &mut tape_len, max_tape) {
                    Some(ptr) => head_pointer = ptr,
                    None => {
                        println!(
                            "reached max memory {} cells; stopping at {} steps",
                            (max_tape - 2).separate_with_commas(),
                            step_count.separate_with_commas()
                        );
                        let elapsed_seconds = start_time.elapsed().as_secs_f64();
                        println!("{:.3} s", elapsed_seconds);
                        return Ok(Summary {
                            step_count,
                            final_state_id: state_id,
                            run_termination: RunTermination::MaxMemoryLeft,
                            elapsed_seconds,
                            tape_len,
                        });
                    }
                }
            } else if hit_right {
                match extend_tape_right(&mut tape, &mut tape_len, max_tape) {
                    Some(ptr) => head_pointer = ptr,
                    None => {
                        println!(
                            "reached max memory {} cells; stopping at {} steps",
                            (max_tape - 2).separate_with_commas(),
                            step_count.separate_with_commas()
                        );
                        let elapsed_seconds = start_time.elapsed().as_secs_f64();
                        println!("{:.3} s", elapsed_seconds);
                        return Ok(Summary {
                            step_count,
                            final_state_id: state_id,
                            run_termination: RunTermination::MaxMemoryRight,
                            elapsed_seconds,
                            tape_len,
                        });
                    }
                }
            } else {
                panic!("unexpected boundary pointer returned from asm");
            }
            if step_count >= next_report {
                let crossed = (step_count - next_report) / interval + 1;
                let last = next_report + (crossed - 1) * interval.get();
                let elapsed = start_time.elapsed().as_secs_f64();
                let done = step_count as f64;
                let steps_per_sec = if elapsed > 0.0 { done / elapsed } else { 0.0 };
                let remaining = (max_steps.saturating_sub(step_count)) as f64;
                let eta = if steps_per_sec > 0.0 {
                    remaining / steps_per_sec
                } else {
                    f64::INFINITY
                };
                let total = if eta.is_finite() {
                    format_duration(elapsed + eta)
                } else {
                    String::from("--:--:--")
                };
                println!(
                    "{} steps (ETA {:.3} s, total ~ {}, elapsed {:.3} s)",
                    last.separate_with_commas(),
                    total,
                    eta,
                    elapsed
                );
                next_report = last.saturating_add(interval.get());
            }
        }
    }
}
/// Executes up to HEARTBEAT steps starting at `head`.
/// Returns (new_head, status_code, remaining_steps)
/// - `status_code`: 0 = ran HEARTBEAT, 1 = halted, 2 = boundary encountered
/// - remaining_steps: RCX after exit; steps_taken = HEARTBEAT - remaining_steps
unsafe fn bb5_champ_compiled(
    mut head: *mut u8,
    mut state_id: u8,
    heartbeat: u64,
) -> (*mut u8, u8, u64, u8) {
    let mut status_code: u8;
    let steps_taken: u64;
    unsafe {
        core::arch::asm!(
            tm_prog!(BB5,
                (A, 0, (1, R, B), (1, L, C)),
                (B, 1, (1, R, C), (1, R, B)),
                (C, 2, (1, R, D), (0, L, E)),
                (D, 3, (1, L, A), (1, L, D)),
                (E, 4, (1, R, HALT), (0, L, A)),
            ),
            inout("rsi") head,                // head pointer in/out
            lateout("r8") steps_taken,        // steps taken this heartbeat
            lateout("al") status_code,        // status code in AL
            inout("bl") state_id,            // state id in/out
            out("rdx") _,                     // clobber DL container
            out("rcx") _,                     // clobber: RCX used as loop counter
            hb = in(reg) heartbeat,
            options(nostack)
        );
    };
    (head, status_code, steps_taken, state_id)
}

/// BB6 Contender heartbeat using the macro-generated asm template
#[allow(clippy::too_many_lines)]
unsafe fn bb6_contender_compiled(
    mut head: *mut u8,
    mut state_id: u8,
    heartbeat: u64,
) -> (*mut u8, u8, u64, u8) {
    let mut status_code: u8;
    let steps_taken: u64;
    unsafe {
        core::arch::asm!(
            tm_prog!(BB6,
                // 0: A 1RB, B 1RC, C 1LC, D 0LE, E 1LF, F 0RC
                // 1: A 0LD, B 0RF, C 1LA, D 1RH, E 0RB, F 0RE
                (A, 0, (1, R, B), (0, L, D)),
                (B, 1, (1, R, C), (0, R, F)),
                (C, 2, (1, L, C), (1, L, A)),
                (D, 3, (0, L, E), (1, R, HALT)),
                (E, 4, (1, L, F), (0, R, B)),
                (F, 5, (0, R, C), (0, R, E)),
            ),
            inout("rsi") head,
            lateout("r8") steps_taken,
            lateout("al") status_code,
            inout("bl") state_id,
            out("rdx") _,
            out("rcx") _,
            hb = in(reg) heartbeat,
            options(nostack)
        );
    };
    (head, status_code, steps_taken, state_id)
}

fn extend_tape_left(
    tape: &mut Vec<u8>,
    total_len: &mut usize,
    max_total: usize,
) -> Option<*mut u8> {
    let old_total = *total_len;
    let old_interior = old_total.saturating_sub(2);
    let growth = old_interior.max(1);
    let new_total = old_total + growth;
    if new_total > max_total {
        return None;
    }
    let mut new_tape = vec![0u8; new_total];
    new_tape[0] = 2;
    new_tape[new_total - 1] = 2;
    // Copy old interior shifted right by `growth`.
    let dst_start = 1 + growth;
    let dst_end = dst_start + old_interior;
    new_tape[dst_start..dst_end].copy_from_slice(&tape[1..(old_total - 1)]);
    *tape = new_tape;
    *total_len = new_total;
    println!(
        "tape grown LEFT to {} cells",
        (new_total - 2).separate_with_commas()
    );
    // Head should be at the first newly-added interior cell.
    Some(unsafe { tape.as_mut_ptr().add(growth) })
}

fn extend_tape_right(
    tape: &mut Vec<u8>,
    total_len: &mut usize,
    max_total: usize,
) -> Option<*mut u8> {
    let old_total = *total_len;
    let old_interior = old_total.saturating_sub(2);
    let growth = old_interior.max(1);
    let new_total = old_total + growth;
    if new_total > max_total {
        return None;
    }
    let mut new_tape = vec![0u8; new_total];
    new_tape[0] = 2;
    new_tape[new_total - 1] = 2;
    // Copy old interior at the same offset.
    new_tape[1..(1 + old_interior)].copy_from_slice(&tape[1..(old_total - 1)]);
    *tape = new_tape;
    *total_len = new_total;
    println!(
        "tape grown RIGHT to {} cells",
        (new_total - 2).separate_with_commas()
    );
    // Head should be the first newly-added interior cell to the right of the old end.
    Some(unsafe { tape.as_mut_ptr().add(old_total - 1) })
}

#[cfg(test)]
mod tests {
    use std::num::NonZero;

    use super::*;

    #[test]
    fn args_into_config_copies_fields() {
        let args = Args {
            program: ProgramSelect::Bb6Contender,
            interval: NonZero::new(42).unwrap().get(),
            max_steps: 7,
            min_tape: 128,
            max_tape: 256,
        };
        let config: CompiledMachine = args.try_into().expect("conversion should succeed");
        assert_eq!(config.program, ProgramSelect::Bb6Contender);
        assert_eq!(config.interval, NonZero::new(42).unwrap());
        assert_eq!(config.max_steps, 7);
        assert_eq!(config.min_tape, 128);
        assert_eq!(config.max_tape, 256);
    }

    #[test]
    fn run_compiled_machine_errors_on_zero_status_interval() {
        let config = CompiledMachine {
            min_tape: 128,
            interval: NonZero::new(0).unwrap(),
            max_tape: 256,
            max_steps: 1,
            program: ProgramSelect::Bb5Champ,
        };
        match config.run() {
            Err(Error::IntervalTooSmall { interval }) => {
                assert_eq!(interval, 0);
            }
            other => panic!("expected IntervalTooSmall error, got {:?}", other),
        }
    }

    #[test]
    fn run_compiled_machine_errors_on_short_tape() {
        let config = CompiledMachine {
            min_tape: 2,
            interval: NonZero::new(1_000).unwrap(),
            max_tape: 64,
            max_steps: 1,
            program: ProgramSelect::Bb5Champ,
        };
        match config.run() {
            Err(Error::TapeTooShort { min_tape: tape_len }) => {
                assert_eq!(tape_len, 2);
            }
            other => panic!("expected TapeTooShort error, got {:?}", other),
        }
    }

    #[test]
    fn run_compiled_machine_errors_on_small_max_tape() {
        let config = CompiledMachine {
            min_tape: 4,
            interval: NonZero::new(1_000).unwrap(),
            max_tape: 2,
            max_steps: 1,
            program: ProgramSelect::Bb5Champ,
        };
        match config.run() {
            Err(Error::MaxTapeTooSmall { max_tape }) => assert_eq!(max_tape, 2),
            other => panic!("expected MaxTapeTooSmall error, got {:?}", other),
        }
    }

    #[test]
    fn run_compiled_machine_clamps_tape_len() {
        let summary = CompiledMachine {
            min_tape: 128,
            interval: NonZero::new(1_000).unwrap(),
            max_tape: 64,
            max_steps: 1,
            program: ProgramSelect::Bb5Champ,
        }
        .run()
        .expect("config should clamp");
        assert_eq!(summary.tape_len, 64);
    }

    #[test]
    fn run_compiled_machine_stops_at_max_steps() {
        let summary = CompiledMachine {
            min_tape: 128,
            interval: NonZero::new(1_000_000).unwrap(),
            max_tape: 1usize << 16,
            max_steps: 1_000,
            program: ProgramSelect::Bb5Champ,
        }
        .run()
        .expect("run should succeed");
        assert_eq!(summary.step_count, 1_000);
        assert_eq!(summary.run_termination, RunTermination::MaxSteps);
        assert!(summary.final_state_id <= 4);
        assert!(summary.elapsed_seconds >= 0.0);
        assert!(summary.tape_len >= 3);
    }
}
