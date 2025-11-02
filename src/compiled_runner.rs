#![allow(named_asm_labels)]

use crate::{asmline, s, tm_dispatch, tm_move, tm_next, tm_prog, tm_state_block, tm_store_on_0, tm_store_on_1};
use derive_more::{Display, Error};
use std::{num::NonZeroU64, time::Instant};
use thousands::Separable;

const DEFAULT_MIN_TAPE: usize = 2_097_152;
const DEFAULT_MAX_TAPE: usize = 16_777_216;
const DEFAULT_INTERVAL: u64 = 10_000_000;

/// Identifies which compiled Turing machine to run
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum CompiledFnId {
    Bb5Champ,
    Bb6Contender,
    Bb33_355K,
}

impl CompiledFnId {
    fn compiled_fn(self) -> CompiledFn {
        match self {
            CompiledFnId::Bb5Champ => bb5_champ_compiled,
            CompiledFnId::Bb6Contender => bb6_contender_compiled,
            CompiledFnId::Bb33_355K => bb33_355_k_compiled,
        }
    }
}

/// Configuration errors
#[derive(Debug, Display, Error)]
pub enum ConfigError {
    #[display("min_tape must be >= 3 (two sentinels + at least one interior); got {min_tape}")]
    TapeTooShort { min_tape: usize },
    #[display("max_tape must be >= 3 (two sentinels + at least one interior); got {max_tape}")]
    MaxTapeTooSmall { max_tape: usize },
    #[display("interval must be > 0; got {interval}")]
    IntervalTooSmall { interval: u64 },
}

/// Configuration for running a compiled Turing machine
#[derive(Debug, Clone)]
pub struct Config {
    pub min_tape: usize,
    pub interval: NonZeroU64,
    pub max_tape: usize,
    pub max_steps: u64,
    compiled_fn: CompiledFn,
    quiet: bool,
}

impl Config {
    pub fn new(
        compiled_fn_id: CompiledFnId,
        interval: u64,
        max_steps: u64,
        min_tape: usize,
        max_tape: usize,
    ) -> Result<Self, ConfigError> {
        if min_tape < 3 {
            return Err(ConfigError::TapeTooShort { min_tape });
        }
        if max_tape < min_tape {
            return Err(ConfigError::MaxTapeTooSmall { max_tape });
        }
        let interval = NonZeroU64::new(interval).ok_or(ConfigError::IntervalTooSmall { interval })?;
        let compiled_fn = compiled_fn_id.compiled_fn();
        Ok(Self {
            min_tape,
            interval,
            max_tape,
            max_steps,
            compiled_fn,
            quiet: false,
        })
    }

    #[must_use]
    pub fn with_quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    pub fn run(self) -> Summary {
        RuntimeState::new(&self).run()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    OkChunk,
    Halted,
    Boundary,
}

type CompiledFn = unsafe fn(&mut RuntimeState<'_>) -> Status;

/// How the machine run terminated
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunTermination {
    Halted,
    MaxSteps,
    MaxMemoryLeft,
    MaxMemoryRight,
}

/// Summary of a completed machine run
#[derive(Debug, Clone)]
pub struct Summary {
    pub step_count: u64,
    pub state_index: u8,
    pub run_termination: RunTermination,
    pub elapsed_secs: f64,
    tape: Vec<u8>,
    origin_index: usize,
}

impl Summary {
    /// Returns a view of the interior tape, excluding boundary sentinels
    pub fn tape(&self) -> &[u8] {
        debug_assert!(self.tape.len() >= 3);
        &self.tape[1..self.tape.len() - 1]
    }

    /// Origin index within the interior slice returned by `tape()`
    pub fn origin_index(&self) -> usize {
        debug_assert!(self.origin_index >= 1);
        self.origin_index - 1
    }
}

#[derive(Debug)]
struct RuntimeState<'a> {
    config: &'a Config,
    tape: Vec<u8>,
    head_pointer: *mut u8,
    state_index: u8,
    report_at_step: u64,
    step_count: u64,
    start_time: Instant,
    origin_index: usize,
}

impl<'a> RuntimeState<'a> {
    fn new(config: &'a Config) -> Self {
        let mut tape: Vec<u8> = vec![0; config.min_tape];
        tape[0] = 2;
        *tape.last_mut().unwrap() = 2;
        let middle = tape.len() >> 1;
        let head_pointer = unsafe { tape.as_mut_ptr().add(middle) };
        let report_at_step = config.interval.get();
        let step_count = 0u64;
        assert!(step_count < report_at_step, "real assert");
        Self {
            config,
            tape,
            head_pointer,
            state_index: 0,
            report_at_step,
            step_count,
            start_time: Instant::now(),
            origin_index: middle,
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

    fn step(&mut self) -> Option<Summary> {
        match unsafe { (self.config.compiled_fn)(self) } {
            Status::OkChunk => self.on_ok_chunk(),
            Status::Halted => Some(self.on_halted()),
            Status::Boundary => self.on_boundary(),
        }
    }

    fn on_ok_chunk(&mut self) -> Option<Summary> {
        assert!(self.step_count <= self.config.max_steps, "real assert");
        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        if self.step_count == self.config.max_steps {
            if !self.config.quiet {
                println!(
                    "reached max steps {}; stopping",
                    self.step_count.separate_with_commas()
                );
                println!("{:.3} s", elapsed_secs);
            }
            use std::mem;
            let full_tape = mem::take(&mut self.tape);
            assert!(full_tape.len() >= 3);
            return Some(Summary {
                step_count: self.step_count,
                state_index: self.state_index,
                run_termination: RunTermination::MaxSteps,
                elapsed_secs,
                tape: full_tape,
                origin_index: self.origin_index,
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
        if !self.config.quiet {
            println!(
                "{} steps ({} steps/s, ETA {:.3} s, total ~ {}, elapsed {:.3} s)",
                self.step_count.separate_with_commas(),
                format_steps_per_sec(steps_per_sec),
                eta,
                total,
                elapsed_secs
            );
        }
        assert!(self.report_at_step <= u64::MAX - self.config.interval.get());
        self.report_at_step += self.config.interval.get();
        None
    }

    fn on_halted(&mut self) -> Summary {
        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        if !self.config.quiet {
            println!(
                "halted after {} steps",
                self.step_count.separate_with_commas()
            );
            println!("{:.3} s", elapsed_secs);
        }
        let full_tape = std::mem::take(&mut self.tape);
        assert!(full_tape.len() >= 3);
        Summary {
            step_count: self.step_count,
            state_index: self.state_index,
            run_termination: RunTermination::Halted,
            elapsed_secs,
            tape: full_tape,
            origin_index: self.origin_index,
        }
    }

    fn on_boundary(&mut self) -> Option<Summary> {
        assert!(self.step_count < self.config.max_steps, "real assert");
        assert!(self.tape.len() >= 3, "real assert");
        if self.head_pointer.cast_const() == self.tape.as_ptr() {
            match self.extend_tape_left() {
                Some(ptr) => self.head_pointer = ptr,
                None => return Some(self.max_mem_summary(RunTermination::MaxMemoryLeft)),
            }
        } else {
            let right = unsafe { self.tape.as_ptr().add(self.tape.len() - 1) };
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

    fn max_mem_summary(&mut self, side: RunTermination) -> Summary {
        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        if !self.config.quiet {
            println!(
                "reached max memory {} cells; stopping at {} steps",
                (self.config.max_tape - 2).separate_with_commas(),
                self.step_count.separate_with_commas()
            );
            println!("{:.3} s", elapsed_secs);
        }
        let full_tape = std::mem::take(&mut self.tape);
        assert!(full_tape.len() >= 3);
        Summary {
            step_count: self.step_count,
            state_index: self.state_index,
            run_termination: side,
            elapsed_secs,
            tape: full_tape,
            origin_index: self.origin_index,
        }
    }

    fn extend_tape_left(&mut self) -> Option<*mut u8> {
        assert!(self.tape.len() >= 3, "real assert");
        let old_interior = self.tape.len() - 2;
        let new_total = old_interior * 2 + 2;
        if new_total > self.config.max_tape {
            return None;
        }
        let mut new_tape = vec![0u8; new_total];
        new_tape[0] = 2;
        *new_tape.last_mut().unwrap() = 2;
        let dst_start = 1 + old_interior;
        let dst_end = dst_start + old_interior;
        new_tape[dst_start..dst_end].copy_from_slice(&self.tape[1..(old_interior + 1)]);
        self.tape = new_tape;
        if !self.config.quiet {
            println!(
                "tape grown LEFT to {} cells",
                (new_total - 2).separate_with_commas()
            );
        }
        self.origin_index += old_interior;
        Some(unsafe { self.tape.as_mut_ptr().add(old_interior) })
    }

    fn extend_tape_right(&mut self) -> Option<*mut u8> {
        assert!(self.tape.len() >= 3, "real assert");
        let old_total = self.tape.len();
        let old_interior = old_total - 2;
        let new_total = old_total + old_interior;
        if new_total > self.config.max_tape {
            return None;
        }
        let old_right = old_total - 1;
        let mut new_tape = vec![0u8; new_total];
        new_tape[0] = 2;
        *new_tape.last_mut().unwrap() = 2;
        new_tape[1..(1 + old_interior)].copy_from_slice(&self.tape[1..(old_total - 1)]);
        self.tape = new_tape;
        if !self.config.quiet {
            println!(
                "tape grown RIGHT to {} cells",
                (new_total - 2).separate_with_commas()
            );
        }
        Some(unsafe { self.tape.as_mut_ptr().add(old_right) })
    }
}

fn format_steps_per_sec(steps_per_sec: f64) -> String {
    let value = if steps_per_sec.is_finite() {
        steps_per_sec.max(0.0).round() as u64
    } else {
        0
    };
    value.separate_with_commas()
}

fn format_duration(total_secs: f64) -> String {
    if !total_secs.is_finite() {
        return String::from("--:--:--");
    }
    let clamped = total_secs.max(0.0);
    let duration = std::time::Duration::from_secs_f64(clamped);
    humantime::format_duration(duration).to_string()
}

// Legacy API for backward compatibility
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompiledProgram {
    Bb5Champ,
    Bb6Contender,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunSummary {
    pub step_count: u64,
    pub nonzero_count: u32,
    pub termination: RunTermination,
}

pub fn run_compiled_program(program: CompiledProgram, max_steps: u64) -> RunSummary {
    let compiled_fn_id = match program {
        CompiledProgram::Bb5Champ => CompiledFnId::Bb5Champ,
        CompiledProgram::Bb6Contender => CompiledFnId::Bb6Contender,
    };

    let config = Config::new(
        compiled_fn_id,
        DEFAULT_INTERVAL,
        max_steps,
        DEFAULT_MIN_TAPE,
        DEFAULT_MAX_TAPE,
    )
    .expect("default config should be valid")
    .with_quiet(true);

    let summary = config.run();
    let nonzeros = summary.tape().iter().filter(|&&v| v != 0).count();

    RunSummary {
        step_count: summary.step_count,
        nonzero_count: u32::try_from(nonzeros).expect("nonzero count fits in u32"),
        termination: summary.run_termination,
    }
}

macro_rules! define_compiled_stepper {
    ($fn_name:ident, $( $state:tt ),+ $(,)? ) => {
        unsafe fn $fn_name(runtime_state: &mut RuntimeState<'_>) -> Status {
            let mut head = runtime_state.head_pointer;
            let mut state = runtime_state.state_index;
            let limit = runtime_state.config.max_steps.min(runtime_state.report_at_step);
            assert!(limit > runtime_state.step_count, "runtime_state must have credit");
            let heartbeat = limit - runtime_state.step_count;
            let mut status_code: u8;
            let mut steps_taken: u64;
            // Safety: the inline assembly steppers maintain the same invariants as the
            // hand-written interpreter: the head pointer stays within the allocated tape,
            // clobbers are declared, and the stack is untouched (`options(nostack)`).
            unsafe {
                core::arch::asm!(
                    "mov r9, {hb}",
                    tm_prog!($fn_name, $( $state ),+),
                    inout("rsi") head,
                    lateout("r8") steps_taken,
                    lateout("al") status_code,
                    inout("bl") state,
                    out("rdx") _,
                    out("r10") _,
                    out("r9") _,
                    hb = in(reg) heartbeat,
                    options(nostack)
                );
            }
            runtime_state.head_pointer = head;
            runtime_state.state_index = state;
            runtime_state.step_count += steps_taken;
            match status_code {
                0 => Status::OkChunk,
                1 => Status::Halted,
                2 => Status::Boundary,
                other => panic!("unexpected status code {other}"),
            }
        }
    }
}

// Unconditional store for any symbol value (0..=254)
#[rustfmt::skip]
macro_rules! tm_store_any {
    ($val:literal) => { asmline!("mov byte ptr [rsi], ", s!($val)) };
}

// 3-symbol (0,1,2) stepper generator for A/B/C states
#[rustfmt::skip]
macro_rules! define_compiled_stepper_3sym {
    ($fn_name:ident,
        ($A:ident, $idA:expr, ($a0w:literal, $a0d:ident, $a0n:ident), ($a1w:literal, $a1d:ident, $a1n:ident), ($a2w:literal, $a2d:ident, $a2n:ident)),
        ($B:ident, $idB:expr, ($b0w:literal, $b0d:ident, $b0n:ident), ($b1w:literal, $b1d:ident, $b1n:ident), ($b2w:literal, $b2d:ident, $b2n:ident)),
        ($C:ident, $idC:expr, ($c0w:literal, $c0d:ident, $c0n:ident), ($c1w:literal, $c1d:ident, $c1n:ident), ($c2w:literal, $c2d:ident, $c2n:ident))
        $(,)?
    ) => {
        unsafe fn $fn_name(runtime_state: &mut RuntimeState<'_>) -> Status {
            let mut head_local: *mut u8 = runtime_state.head_pointer;
            let mut state_local: u8 = runtime_state.state_index;
            let step_limit: u64 = runtime_state.config.max_steps.min(runtime_state.report_at_step);
            assert!(step_limit > runtime_state.step_count);
            let heartbeat: u64 = step_limit - runtime_state.step_count;
            let mut status_code: u8;
            let mut steps_taken_local: u64;
            // Safety: Uses x86_64 inline assembly to step the Turing machine.
            unsafe {
                core::arch::asm!(
                    "mov r9, {hb}",
                    asmline!("mov r10, r9"),
                    // dispatch
                    asmline!("cmp bl, ", s!($idA)), asmline!("je ", s!($fn_name), "_", s!($A)),
                    asmline!("cmp bl, ", s!($idB)), asmline!("je ", s!($fn_name), "_", s!($B)),
                asmline!("cmp bl, ", s!($idC)), asmline!("je ", s!($fn_name), "_", s!($C)),
                asmline!("jmp ", s!($fn_name), "_", s!($A)),
                // A
                asmline!(s!($fn_name), "_", s!($A), ":"),
                asmline!("mov bl, ", s!($idA)),
                asmline!("mov dl, [rsi]"),
                asmline!("cmp dl, 255"), asmline!("je ", s!($fn_name), "_BOUNDARY_", s!($A)),
                asmline!("cmp dl, 0"), asmline!("je ", s!($fn_name), "_", s!($A), "_ZERO"),
                asmline!("cmp dl, 1"), asmline!("je ", s!($fn_name), "_", s!($A), "_ONE"),
                asmline!("jmp ", s!($fn_name), "_", s!($A), "_TWO"),
                asmline!(s!($fn_name), "_", s!($A), "_ZERO:"), tm_store_any!($a0w), tm_move!($a0d), tm_next!($fn_name, $a0n, $idA),
                asmline!(s!($fn_name), "_", s!($A), "_ONE:"),  tm_store_any!($a1w), tm_move!($a1d), tm_next!($fn_name, $a1n, $idA),
                asmline!(s!($fn_name), "_", s!($A), "_TWO:"),  tm_store_any!($a2w), tm_move!($a2d), tm_next!($fn_name, $a2n, $idA),
                asmline!(s!($fn_name), "_BOUNDARY_", s!($A), ":"), asmline!("mov bl, ", s!($idA)), asmline!("jmp ", s!($fn_name), "_BOUNDARY"),
                // B
                asmline!(s!($fn_name), "_", s!($B), ":"),
                asmline!("mov bl, ", s!($idB)),
                asmline!("mov dl, [rsi]"),
                asmline!("cmp dl, 255"), asmline!("je ", s!($fn_name), "_BOUNDARY_", s!($B)),
                asmline!("cmp dl, 0"), asmline!("je ", s!($fn_name), "_", s!($B), "_ZERO"),
                asmline!("cmp dl, 1"), asmline!("je ", s!($fn_name), "_", s!($B), "_ONE"),
                asmline!("jmp ", s!($fn_name), "_", s!($B), "_TWO"),
                asmline!(s!($fn_name), "_", s!($B), "_ZERO:"), tm_store_any!($b0w), tm_move!($b0d), tm_next!($fn_name, $b0n, $idB),
                asmline!(s!($fn_name), "_", s!($B), "_ONE:"),  tm_store_any!($b1w), tm_move!($b1d), tm_next!($fn_name, $b1n, $idB),
                asmline!(s!($fn_name), "_", s!($B), "_TWO:"),  tm_store_any!($b2w), tm_move!($b2d), tm_next!($fn_name, $b2n, $idB),
                asmline!(s!($fn_name), "_BOUNDARY_", s!($B), ":"), asmline!("mov bl, ", s!($idB)), asmline!("jmp ", s!($fn_name), "_BOUNDARY"),
                // C
                asmline!(s!($fn_name), "_", s!($C), ":"),
                asmline!("mov bl, ", s!($idC)),
                asmline!("mov dl, [rsi]"),
                asmline!("cmp dl, 255"), asmline!("je ", s!($fn_name), "_BOUNDARY_", s!($C)),
                asmline!("cmp dl, 0"), asmline!("je ", s!($fn_name), "_", s!($C), "_ZERO"),
                asmline!("cmp dl, 1"), asmline!("je ", s!($fn_name), "_", s!($C), "_ONE"),
                asmline!("jmp ", s!($fn_name), "_", s!($C), "_TWO"),
                asmline!(s!($fn_name), "_", s!($C), "_ZERO:"), tm_store_any!($c0w), tm_move!($c0d), tm_next!($fn_name, $c0n, $idC),
                asmline!(s!($fn_name), "_", s!($C), "_ONE:"),  tm_store_any!($c1w), tm_move!($c1d), tm_next!($fn_name, $c1n, $idC),
                asmline!(s!($fn_name), "_", s!($C), "_TWO:"),  tm_store_any!($c2w), tm_move!($c2d), tm_next!($fn_name, $c2n, $idC),
                asmline!(s!($fn_name), "_BOUNDARY_", s!($C), ":"), asmline!("mov bl, ", s!($idC)), asmline!("jmp ", s!($fn_name), "_BOUNDARY"),
                // end
                asmline!(s!($fn_name), "_BOUNDARY:"), asmline!("mov al, 2"),
                asmline!(s!($fn_name), "_END:"), asmline!("mov r8, r9"), asmline!("sub r8, r10"),
                inout("rsi") head_local,
                lateout("r8") steps_taken_local,
                lateout("al") status_code,
                inout("bl") state_local,
                out("rdx") _, out("r10") _, out("r9") _,
                hb = in(reg) heartbeat,
                options(nostack)
                );
            }
            runtime_state.head_pointer = head_local;
            runtime_state.state_index = state_local;
            runtime_state.step_count += steps_taken_local;
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

// BB(3,3) champion (~355,317 steps) spec: 1RB 2LA 1RA_1LA 1RZ 1RC_2RB 1RC 2RB
define_compiled_stepper_3sym!(
    bb33_355_k_compiled,
    (A, 0, (1, R, B), (2, L, A), (1, R, A)),
    (B, 1, (1, L, A), (1, R, HALT), (1, R, C)),
    (C, 2, (2, R, B), (1, R, C), (2, R, B)),
);
