// cmk notes
// This is good. To refine the API and structure further, need to know our use case more, e.g. using it in
// the larger program.
// Other things to do:
// -- support more symbols
// -- use derive macros to make the use of the asm template more ergonomic.
//           we might be able to more optimizations, e.g. if both 0/1 branch write same value or move same dir.
// -- when used elsewhere likely want to return the final tape contents, not just step count.
// -- haven't carefully review the tape growth code.

#![allow(named_asm_labels)] // Allow alphabetic labels in inline asm for readability
use clap::{Parser, ValueEnum};
use derive_more::{Display, Error};
use std::{
    num::NonZeroU64,
    time::{Duration, Instant},
};

fn format_steps_per_sec(steps_per_sec: f64) -> String {
    let value = if steps_per_sec.is_finite() {
        steps_per_sec.max(0.0).round() as u64
    } else {
        0
    };
    value.separate_with_commas()
}
use thousands::Separable;

// Macro helpers to generate the full asm template from a TM spec
macro_rules! tm_move {
    (R) => {
        /* move head Right */
        "add rsi, 1\n"
    };
    (L) => {
        /* move head Left */
        "sub rsi, 1\n"
    };
}

// one line of assembly -> compile-time string with trailing '\n'
macro_rules! asmline {
    ($($p:expr),* $(,)?) => { concat!($($p),*, "\n") };
}

macro_rules! s {
    ($x:expr) => {
        stringify!($x)
    };
}

// Skip redundant stores based on the read branch.
// - On the 0-branch (read 0), writing 0 is a no-op; only write if 1.
// - On the 1-branch (read 1), writing 1 is a no-op; only write if 0.
macro_rules! tm_store_on_0 {
    (0) => {
        ""
    };
    (1) => {
        asmline!("mov byte ptr [rsi], 1")
    };
    ($other:tt) => {
        asmline!("mov byte ptr [rsi], ", s!($other))
    };
}

macro_rules! tm_store_on_1 {
    (0) => {
        asmline!("mov byte ptr [rsi], 0")
    };
    (1) => {
        ""
    };
    ($other:tt) => {
        asmline!("mov byte ptr [rsi], ", s!($other))
    };
}

#[rustfmt::skip] // keep comment alignment, don't reflow
macro_rules! tm_next {
    // HALT case
    ($P:ident, HALT, $id:expr) => {
        concat!(
            /* consume one step           */ asmline!("dec r10"),
            /* set status = halt          */ asmline!("mov al, 1"),
            /* record current state id    */ asmline!("mov bl, ", s!($id)),
            /* jump to end label          */ asmline!("jmp ", s!($P), "_END"),
        )
    };

    // goto next state
    ($P:ident, $N:ident, $id:expr) => {
        concat!(
            /* consume one step           */ asmline!("dec r10"),
            /* out of credit?             */ asmline!("jz ", s!($P), "_END"),
            /* goto next state            */ asmline!("jmp ", s!($P), "_", s!($N)),
        )
    };
}

#[rustfmt::skip] // keep comment alignment, don't reflow
macro_rules! tm_dispatch {
    ( $P:ident, $S:ident, $id:expr ) => {
        concat!(
            /* compare BL with state id */ asmline!("cmp bl, ", s!($id)),
            /* if equal, jump to state */ asmline!("je ", s!($P), "_", s!($S)),
        )
    };
}
#[rustfmt::skip] // keep comment alignment, don't reflow
macro_rules! tm_state_block {
    ( $P:ident, $S:ident, $id:expr, ( $w0:literal, $d0:ident, $n0:ident ), ( $w1:literal, $d1:ident, $n1:ident ) ) => {
        concat!(
            /* state label                 */ asmline!(s!($P), "_", s!($S), ":"),
            /* persist current state id    */ asmline!("mov bl, ", s!($id)),
            /* load cell                   */ asmline!("mov dl, [rsi]"),
            // cmk000 The boundary check is virtually free because of branch prediction
            /* boundary sentinel?          */ asmline!("cmp dl, 2"),
            /* jump if boundary            */ asmline!("je ", s!($P), "_BOUNDARY_", s!($S)),
            // cmk000 end
            /* is cell == 0?               */ asmline!("test dl, dl"),
            /* branch if 1                 */ asmline!("jnz ", s!($P), "_", s!($S), "_ONE"),
            /* write on 0-branch           */ tm_store_on_0!($w0),
            tm_move!($d0),
            /* jump to next (0-branch)     */ tm_next!($P, $n0, $id),
            /* 1-branch label              */ asmline!(s!($P), "_", s!($S), "_ONE:"),
            /* write on 1-branch           */ tm_store_on_1!($w1),
            tm_move!($d1),
            /* jump to next (1-branch)     */ tm_next!($P, $n1, $id),
            /* boundary label              */ asmline!(s!($P), "_BOUNDARY_", s!($S), ":"),
            /* record state id             */ asmline!("mov bl, ", s!($id)),
            /* go to common boundary       */ asmline!("jmp ", s!($P), "_BOUNDARY"),
        )
    };
}

#[rustfmt::skip] // keep comment alignment, don't reflow
macro_rules! tm_prog {
    ( $P:ident, ($S0:ident, $id0:expr, $z0:tt, $o0:tt) $(, ($S:ident, $id:expr, $z:tt, $o:tt) )* $(,)? ) => {
        concat!(
            /* clear status (AL)            */ asmline!("xor eax, eax"),
            /* set heartbeat (credit) in R10 */ asmline!("mov r10, r9"),
            tm_dispatch!($P, $S0, $id0),
            $( tm_dispatch!($P, $S, $id), )*
            /* jump to first state          */ asmline!("jmp ", s!($P), "_", s!($S0)),
            tm_state_block!($P, $S0, $id0, $z0, $o0),
            $( tm_state_block!($P, $S, $id, $z, $o), )*
            /* common boundary label        */ asmline!(s!($P), "_BOUNDARY:"),
            /* set status = boundary        */ asmline!("mov al, 2"),
            /* chunk end                    */ asmline!(s!($P), "_END:"),
            /* copy heartbeat to r8         */ asmline!("mov r8, r9"),
            /* steps_taken = hb - r10       */ asmline!("sub r8, r10"),
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
    #[arg(long, value_enum, default_value_t = CompiledFnId::Bb5Champ)]
    program: CompiledFnId,

    /// Status print interval in steps. Omit to disable periodic reports.
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum CompiledFnId {
    Bb5Champ,
    Bb6Contender,
}

impl CompiledFnId {
    fn compiled_fn(self) -> CompiledFn {
        match self {
            CompiledFnId::Bb5Champ => bb5_champ_compiled,
            CompiledFnId::Bb6Contender => bb6_contender_compiled,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    OkChunk,
    Halted,
    Boundary,
}

// Compiled function type: operates on RuntimeState state.
// Must update head_pointer/state_index and add executed steps to step_count,
// respecting runtime_state.max_steps and runtime_state.report_at_step.
type CompiledFn = unsafe fn(&mut RuntimeState<'_>) -> Status;

#[derive(Debug, Clone)]
pub struct Config {
    pub min_tape: usize,
    pub interval: NonZeroU64,
    pub max_tape: usize,
    pub max_steps: u64,
    compiled_fn: CompiledFn,
}

impl TryFrom<Args> for Config {
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
        Self::new(program, interval, max_steps, min_tape, max_tape)
    }
}

impl Config {
    pub fn new(
        compiled_fn_id: CompiledFnId,
        interval: u64,
        max_steps: u64,
        min_tape: usize,
        max_tape: usize,
    ) -> Result<Self, Error> {
        if min_tape < 3 {
            return Err(Error::TapeTooShort { min_tape });
        }
        if max_tape < min_tape {
            return Err(Error::MaxTapeTooSmall { max_tape });
        }
        let interval = NonZeroU64::new(interval).ok_or(Error::IntervalTooSmall { interval })?;
        let compiled_fn = compiled_fn_id.compiled_fn();
        Ok(Self {
            min_tape,
            interval,
            max_tape,
            max_steps,
            compiled_fn,
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
    pub state_index: u8,
    pub run_termination: RunTermination,
    pub elapsed_secs: f64,
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
    let config: Config = Args::parse().try_into().unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let _summary = config.run();
}

impl Config {
    pub fn run(self) -> Summary {
        RuntimeState::new(&self).run()
    }
}

#[derive(Debug)]
struct RuntimeState<'a> {
    // configuration (borrowed)
    config: &'a Config,
    // runtime state
    tape: Vec<u8>,
    tape_len: usize,
    head_pointer: *mut u8,
    state_index: u8,
    report_at_step: u64,
    step_count: u64,
    start_time: Instant,
}

impl<'a> RuntimeState<'a> {
    fn new(config: &'a Config) -> Self {
        let tape_len = config.min_tape;
        let mut tape: Vec<u8> = vec![0; tape_len];
        tape[0] = 2;
        tape[tape_len - 1] = 2;
        let head_pointer = unsafe { tape.as_mut_ptr().add(tape_len >> 1) };
        let report_at_step = config.interval.get();
        let step_count = 0u64;
        assert!(step_count < report_at_step, "real assert");
        Self {
            config,
            tape,
            tape_len,
            head_pointer,
            state_index: 0,
            report_at_step,
            step_count,
            start_time: Instant::now(),
        }
    }

    fn on_ok_chunk(&mut self) -> Option<Summary> {
        assert!(self.step_count <= self.config.max_steps, "real assert");
        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        if self.step_count == self.config.max_steps {
            println!(
                "reached max steps {}; stopping",
                self.step_count.separate_with_commas()
            );
            println!("{:.3} s", elapsed_secs);
            return Some(Summary {
                step_count: self.step_count,
                state_index: self.state_index,
                run_termination: RunTermination::MaxSteps,
                elapsed_secs,
                tape_len: self.tape_len,
            });
        }
        assert!(self.step_count == self.report_at_step, "real assert");
        let (steps_per_sec, eta, total) = if elapsed_secs > 0.0 {
            let steps_per_sec = (self.step_count as f64) / elapsed_secs;
            let eta = ((self.config.max_steps - self.step_count) as f64) / steps_per_sec;
            let total = format_duration(elapsed_secs + eta);
            (steps_per_sec, eta, total)
        } else {
            (0.0, f64::INFINITY, String::from("--:--:--"))
        };
        println!(
            "{} steps ({} steps/s, ETA {:.3} s, total ~ {}, elapsed {:.3} s)",
            self.step_count.separate_with_commas(),
            format_steps_per_sec(steps_per_sec),
            eta,
            total,
            elapsed_secs
        );
        assert!(self.report_at_step <= u64::MAX - self.config.interval.get());
        self.report_at_step += self.config.interval.get();
        None
    }

    fn step(&mut self) -> Option<Summary> {
        match unsafe { (self.config.compiled_fn)(self) } {
            Status::OkChunk => self.on_ok_chunk(),
            Status::Halted => Some(self.on_halted()),
            Status::Boundary => self.on_boundary(),
        }
    }

    fn run(mut self) -> Summary {
        loop {
            assert!(self.step_count < self.config.max_steps, "real assert");
            if let Some(summary) = self.step() {
                return summary;
            }
        }
    }

    fn on_halted(&self) -> Summary {
        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        println!(
            "halted after {} steps",
            self.step_count.separate_with_commas()
        );
        println!("{:.3} s", elapsed_secs);
        Summary {
            step_count: self.step_count,
            state_index: self.state_index,
            run_termination: RunTermination::Halted,
            elapsed_secs,
            tape_len: self.tape_len,
        }
    }

    fn on_boundary(&mut self) -> Option<Summary> {
        assert!(self.step_count < self.config.max_steps, "real assert");
        assert!(self.tape_len >= 3, "real assert");
        if self.head_pointer.cast_const() == self.tape.as_ptr() {
            match self.extend_tape_left() {
                Some(ptr) => self.head_pointer = ptr,
                None => return Some(self.max_mem_summary(RunTermination::MaxMemoryLeft)),
            }
        } else {
            let right = unsafe { self.tape.as_ptr().add(self.tape_len - 1) };
            assert!(
                self.head_pointer.cast_const() == right,
                "real assert: boundary else-branch must be at right sentinel"
            );
            match self.extend_tape_right() {
                Some(ptr) => self.head_pointer = ptr,
                None => return Some(self.max_mem_summary(RunTermination::MaxMemoryRight)),
            }
        }
        None
    }

    fn max_mem_summary(&self, side: RunTermination) -> Summary {
        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        println!(
            "reached max memory {} cells; stopping at {} steps",
            (self.config.max_tape - 2).separate_with_commas(),
            self.step_count.separate_with_commas()
        );
        println!("{:.3} s", elapsed_secs);
        Summary {
            step_count: self.step_count,
            state_index: self.state_index,
            run_termination: side,
            elapsed_secs,
            tape_len: self.tape_len,
        }
    }

    fn extend_tape_left(&mut self) -> Option<*mut u8> {
        let tape = &mut self.tape;
        let total_len = &mut self.tape_len;
        let old_total = *total_len;
        let old_interior = old_total.saturating_sub(2);
        let growth = old_interior.max(1);
        let new_total = old_total + growth;
        if new_total > self.config.max_tape {
            return None;
        }
        let mut new_tape = vec![0u8; new_total];
        new_tape[0] = 2;
        new_tape[new_total - 1] = 2;
        let dst_start = 1 + growth;
        let dst_end = dst_start + old_interior;
        new_tape[dst_start..dst_end].copy_from_slice(&tape[1..(old_total - 1)]);
        *tape = new_tape;
        *total_len = new_total;
        println!(
            "tape grown LEFT to {} cells",
            (new_total - 2).separate_with_commas()
        );
        Some(unsafe { tape.as_mut_ptr().add(growth) })
    }

    fn extend_tape_right(&mut self) -> Option<*mut u8> {
        let tape = &mut self.tape;
        let total_len = &mut self.tape_len;
        let old_total = *total_len;
        let old_interior = old_total.saturating_sub(2);
        let growth = old_interior.max(1);
        let new_total = old_total + growth;
        if new_total > self.config.max_tape {
            return None;
        }
        let mut new_tape = vec![0u8; new_total];
        new_tape[0] = 2;
        new_tape[new_total - 1] = 2;
        new_tape[1..(1 + old_interior)].copy_from_slice(&tape[1..(old_total - 1)]);
        *tape = new_tape;
        *total_len = new_total;
        println!(
            "tape grown RIGHT to {} cells",
            (new_total - 2).separate_with_commas()
        );
        Some(unsafe { tape.as_mut_ptr().add(old_total - 1) })
    }
}
/// Executes up to `*step_budget_in_out` steps starting at `*head_in_out`.
/// Mutates inputs in place and returns `status_code`:
/// - `status_code`: 0 = ran heartbeat fully, 1 = halted, 2 = boundary encountered
/// - `*head_in_out`: updated head pointer
/// - `*state_index_in_out`: updated state index
/// - `*step_budget_in_out`: overwritten with steps taken during this heartbeat
///
/// Safety: Uses inline assembly and relies on `head_in_out` pointing to a valid
/// tape cell within an allocation that includes boundary sentinels. The asm
/// only reads/writes the current cell and moves within the allocation; it also
/// preserves alignment/stack and does not touch memory beyond the tape.
// Shared generator for compiled steppers differing only by program spec
macro_rules! define_compiled_stepper {
    ($fn_name:ident, $( $prog_spec:tt )+ ) => {
        unsafe fn $fn_name(runner: &mut RuntimeState<'_>) -> Status {
            let mut head_local: *mut u8 = runner.head_pointer;
            let mut state_local: u8 = runner.state_index;
            let step_limit: u64 = runner.config.max_steps.min(runner.report_at_step);
            assert!(step_limit > runner.step_count);
            let heartbeat: u64 = step_limit - runner.step_count;
            let mut status_code: u8;
            let mut steps_taken_local: u64;
            unsafe {
                core::arch::asm!(
                    "mov r9, {hb}",
                    tm_prog!($fn_name, $( $prog_spec )+),
                    inout("rsi") head_local,
                    lateout("r8") steps_taken_local,
                    lateout("al") status_code,
                    inout("bl") state_local,
                    out("rdx") _,
                    out("r10") _,
                    out("r9") _,
                    hb = in(reg) heartbeat,
                    options(nostack)
                );
            }
            runner.head_pointer = head_local;
            runner.state_index = state_local;
            runner.step_count += steps_taken_local;
            match status_code {
                0 => Status::OkChunk,
                1 => Status::Halted,
                2 => Status::Boundary,
                other => panic!("unexpected status code from asm: {other}"),
            }
        }
    };
}

define_compiled_stepper!(
    bb5_champ_compiled,
    (A, 0, (1, R, B), (1, L, C)),
    (B, 1, (1, R, C), (1, R, B)),
    (C, 2, (1, R, D), (0, L, E)),
    (D, 3, (1, L, A), (1, L, D)),
    (E, 4, (1, R, HALT), (0, L, A)),
);

define_compiled_stepper!(
    bb6_contender_compiled,
    (A, 0, (1, R, B), (0, L, D)),
    (B, 1, (1, R, C), (0, R, F)),
    (C, 2, (1, L, C), (1, L, A)),
    (D, 3, (0, L, E), (1, R, HALT)),
    (E, 4, (1, L, F), (0, R, B)),
    (F, 5, (0, R, C), (0, R, E)),
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_try_from_copies_fields() {
        let args = Args {
            program: CompiledFnId::Bb6Contender,
            interval: 42,
            max_steps: 7,
            min_tape: 128,
            max_tape: 256,
        };
        let config: Config = args.try_into().expect("conversion should succeed");
        // Verify constructor selected the correct compiled function
        assert_eq!(config.compiled_fn as usize, bb6_contender_compiled as usize);
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
        };
        match Config::try_from(args) {
            Err(Error::IntervalTooSmall { interval }) => assert_eq!(interval, 0),
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
        };
        match Config::try_from(args) {
            Err(Error::TapeTooShort { min_tape }) => assert_eq!(min_tape, 2),
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
        };
        match Config::try_from(args) {
            Err(Error::MaxTapeTooSmall { max_tape }) => assert_eq!(max_tape, 2),
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
        }
        .try_into()
        .expect("conversion should succeed");

        let summary = compiled_machine.run();
        assert_eq!(summary.step_count, 1_000);
        assert_eq!(summary.run_termination, RunTermination::MaxSteps);
        assert!(summary.state_index <= 4);
        assert!(summary.elapsed_secs >= 0.0);
        assert!(summary.tape_len >= 3);
    }

    // This mirrors `cargo run --example compile_machine --release` defaults
    // and asserts the known halting step count for Bb5Champ. It is long-running
    // in debug mode, so we only run it in release builds.
    #[test]
    fn bb5_champ_halts_at_47_million_steps() {
        let compiled_machine: Config = Args {
            program: CompiledFnId::Bb5Champ,
            interval: 20_000_000,
            max_steps: u64::MAX,
            min_tape: 2_097_152,
            max_tape: 16_777_216,
        }
        .try_into()
        .expect("conversion should succeed");

        let summary = compiled_machine.run();
        assert_eq!(summary.run_termination, RunTermination::Halted);
        assert_eq!(summary.step_count, 47_176_870);
    }

    #[test]
    fn bb5_compiled_does_not_halt_early() {
        // Reproduce the bug where resuming at chunk boundaries jumps to a wrong state
        // and halts around ~10,003,798 steps. With the fix, this run should not halt
        // and instead stop at the max_steps limit.
        let compiled_machine: Config = Args {
            program: CompiledFnId::Bb5Champ,
            interval: 10_000_000,
            max_steps: 20_000_000,
            min_tape: 2_097_152,
            max_tape: 16_777_216,
        }
        .try_into()
        .expect("conversion should succeed");

        let summary = compiled_machine.run();
        assert_eq!(summary.run_termination, RunTermination::MaxSteps);
        assert_eq!(summary.step_count, 20_000_000);
        // Should still be in a valid state id (0..=4) for BB5
        assert!(summary.state_index <= 4);
    }

    #[test]
    fn bb5_run() -> Result<(), Error> {
        let summary = Config::new(
            CompiledFnId::Bb5Champ,
            10_000_000,
            u64::MAX,
            2_097_152,
            16_777_216,
        )?
        .run();

        assert_eq!(summary.run_termination, RunTermination::Halted);
        assert_eq!(summary.step_count, 47_176_870);
        assert_eq!(summary.state_index, 4);
        Ok(())
    }
}
