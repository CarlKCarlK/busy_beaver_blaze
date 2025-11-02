#![allow(named_asm_labels)]

use crate::{asmline, s, tm_dispatch, tm_move, tm_next, tm_prog, tm_state_block, tm_store_on_0, tm_store_on_1};

const DEFAULT_MIN_TAPE: usize = 2_097_152;
const DEFAULT_MAX_TAPE: usize = 16_777_216;
const DEFAULT_INTERVAL: u64 = 10_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompiledProgram {
    Bb5Champ,
    Bb6Contender,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunTermination {
    Halted,
    MaxSteps,
    MaxMemoryLeft,
    MaxMemoryRight,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunSummary {
    pub step_count: u64,
    pub nonzero_count: u32,
    pub termination: RunTermination,
}

pub fn run_compiled_program(program: CompiledProgram, max_steps: u64) -> RunSummary {
    let compiled_fn: CompiledFn = match program {
        CompiledProgram::Bb5Champ => bb5_champ_compiled,
        CompiledProgram::Bb6Contender => bb6_contender_compiled,
    };

    let mut runner = Runner::new(
        compiled_fn,
        DEFAULT_MIN_TAPE,
        DEFAULT_MAX_TAPE,
        max_steps,
        DEFAULT_INTERVAL,
    );

    runner.run()
}

type CompiledFn = unsafe fn(&mut Runner) -> Status;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    OkChunk,
    Halted,
    Boundary,
}

struct Runner {
    tape: Vec<u8>,
    head_pointer: *mut u8,
    state_index: u8,
    report_at_step: u64,
    step_count: u64,
    max_steps: u64,
    interval: u64,
    max_tape: usize,
    origin_index: usize,
    compiled_fn: CompiledFn,
}

impl Runner {
    fn new(
        compiled_fn: CompiledFn,
        min_tape: usize,
        max_tape: usize,
        max_steps: u64,
        interval: u64,
    ) -> Self {
        let mut tape = vec![0; min_tape];
        tape[0] = 0xFF;
        *tape.last_mut().expect("tape must be non-empty") = 0xFF;
        let middle = tape.len() / 2;
        let head_pointer = unsafe { tape.as_mut_ptr().add(middle) };
        Self {
            tape,
            head_pointer,
            state_index: 0,
            report_at_step: interval,
            step_count: 0,
            max_steps,
            interval,
            max_tape,
            origin_index: middle,
            compiled_fn,
        }
    }

    fn run(&mut self) -> RunSummary {
        loop {
            if self.step_count >= self.max_steps {
                return self.finish(RunTermination::MaxSteps);
            }
            match unsafe { (self.compiled_fn)(self) } {
                Status::OkChunk => {
                    if self.step_count >= self.max_steps {
                        return self.finish(RunTermination::MaxSteps);
                    }
                    self.report_at_step = self.report_at_step.saturating_add(self.interval);
                }
                Status::Halted => return self.finish(RunTermination::Halted),
                Status::Boundary => {
                    if let Some(summary) = self.on_boundary() {
                        return summary;
                    }
                }
            }
        }
    }

    fn finish(&mut self, termination: RunTermination) -> RunSummary {
        let tape = std::mem::take(&mut self.tape);
        let interior = &tape[1..tape.len() - 1];
        let nonzeros = interior.iter().filter(|&&value| value != 0).count();
        RunSummary {
            step_count: self.step_count,
            nonzero_count: u32::try_from(nonzeros).expect("nonzero count must fit in u32"),
            termination,
        }
    }

    fn on_boundary(&mut self) -> Option<RunSummary> {
        if self.head_pointer.cast_const() == self.tape.as_ptr() {
            match self.extend_tape_left() {
                Some(ptr) => {
                    self.head_pointer = ptr;
                    None
                }
                None => Some(self.finish(RunTermination::MaxMemoryLeft)),
            }
        } else {
            let right = unsafe { self.tape.as_ptr().add(self.tape.len() - 1) };
            if self.head_pointer.cast_const() != right {
                panic!("boundary event without matching sentinel");
            }
            match self.extend_tape_right() {
                Some(ptr) => {
                    self.head_pointer = ptr;
                    None
                }
                None => Some(self.finish(RunTermination::MaxMemoryRight)),
            }
        }
    }

    fn extend_tape_left(&mut self) -> Option<*mut u8> {
        let old_interior = self.tape.len() - 2;
        let new_total = old_interior * 2 + 2;
        if new_total > self.max_tape {
            return None;
        }
        let mut new_tape = vec![0u8; new_total];
        new_tape[0] = 0xFF;
        *new_tape.last_mut().expect("last element exists") = 0xFF;
        let dst_start = 1 + old_interior;
        let dst_end = dst_start + old_interior;
        new_tape[dst_start..dst_end].copy_from_slice(&self.tape[1..(old_interior + 1)]);
        self.tape = new_tape;
        self.origin_index += old_interior;
        Some(unsafe { self.tape.as_mut_ptr().add(old_interior) })
    }

    fn extend_tape_right(&mut self) -> Option<*mut u8> {
        let old_total = self.tape.len();
        let old_interior = old_total - 2;
        let new_total = old_total + old_interior;
        if new_total > self.max_tape {
            return None;
        }
        let old_right = old_total - 1;
        let mut new_tape = vec![0u8; new_total];
        new_tape[0] = 0xFF;
        *new_tape.last_mut().expect("last element exists") = 0xFF;
        new_tape[1..(1 + old_interior)].copy_from_slice(&self.tape[1..(old_total - 1)]);
        self.tape = new_tape;
        Some(unsafe { self.tape.as_mut_ptr().add(old_right) })
    }
}

macro_rules! define_compiled_stepper {
    ($fn_name:ident, $( $state:tt ),+ $(,)? ) => {
        unsafe fn $fn_name(runner: &mut Runner) -> Status {
            let mut head = runner.head_pointer;
            let mut state = runner.state_index;
            let limit = runner.max_steps.min(runner.report_at_step);
            assert!(limit > runner.step_count, "runner must have credit");
            let heartbeat = limit - runner.step_count;
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
            runner.head_pointer = head;
            runner.state_index = state;
            runner.step_count += steps_taken;
            match status_code {
                0 => Status::OkChunk,
                1 => Status::Halted,
                2 => Status::Boundary,
                other => panic!("unexpected status code {other}"),
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
