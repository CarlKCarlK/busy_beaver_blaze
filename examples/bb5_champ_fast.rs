#![allow(named_asm_labels)] // Allow alphabetic labels in inline asm for readability
use clap::Parser;
use thousands::Separable;

// Macro helpers to generate the full asm template from a TM spec
macro_rules! tm_move {
    (R) => {
        "inc rsi\n"
    };
    (L) => {
        "dec rsi\n"
    };
}
macro_rules! tm_next {
    (HALT, $id:expr) => {
        concat!(
            "mov al, 1\n",
            "mov bl, ",
            stringify!($id),
            "\n",
            "jmp END\n"
        )
    };
    ($N:ident, $id:expr) => {
        concat!("jmp ", stringify!($N), "\n")
    };
}
macro_rules! tm_dispatch {
    ($S:ident, $id:expr) => {
        concat!(
            "cmp bl, ",
            stringify!($id),
            "\n",
            "je ",
            stringify!($S),
            "\n"
        )
    };
}
macro_rules! tm_state_block {
    ( $S:ident, $id:expr, ( $w0:literal, $d0:ident, $n0:ident ), ( $w1:literal, $d1:ident, $n1:ident ) ) => {
        concat!(
            stringify!($S),
            ":\n",
            "cmp rcx, 0\n",
            "jne ",
            stringify!($S),
            "_CONT\n",
            "mov bl, ",
            stringify!($id),
            "\n",
            "jmp END\n",
            stringify!($S),
            "_CONT:\n",
            "mov dl, [rsi]\n",
            "cmp dl, 2\n",
            "je BOUNDARY_",
            stringify!($S),
            "\n",
            "test dl, dl\n",
            "jnz ",
            stringify!($S),
            "_ONE\n",
            "mov byte ptr [rsi], ",
            stringify!($w0),
            "\n",
            tm_move!($d0),
            "sub rcx, 1\n",
            tm_next!($n0, $id),
            stringify!($S),
            "_ONE:\n",
            "mov byte ptr [rsi], ",
            stringify!($w1),
            "\n",
            tm_move!($d1),
            "sub rcx, 1\n",
            tm_next!($n1, $id),
            "BOUNDARY_",
            stringify!($S),
            ":\n",
            "mov bl, ",
            stringify!($id),
            "\n",
            "jmp BOUNDARY\n",
        )
    };
}
macro_rules! tm_prog {
    ( ($S0:ident, $id0:expr, $z0:tt, $o0:tt) $(, ($S:ident, $id:expr, $z:tt, $o:tt) )* $(,)? ) => {
        concat!(
            "xor eax, eax\n",
            "mov rcx, {hb}\n",
            tm_dispatch!($S0, $id0),
            $( tm_dispatch!($S, $id), )*
            "jmp ", stringify!($S0), "\n",
            tm_state_block!($S0, $id0, $z0, $o0),
            $( tm_state_block!($S, $id, $z, $o), )*
            "BOUNDARY:\n",
            "mov al, 2\n",
            "END:\n",
            "mov r8, {hb}\n",
            "sub r8, rcx\n",
        )
    };
}

// Note: we intentionally accept only plain numeric values for CLI args now.

// Heartbeat is chosen per-iteration; no fixed default needed.
#[derive(Debug, Parser, Clone)]
#[command(name = "bb5_champ_fast", about = "Fast BB5 runner with inline asm")]
struct Args {
    #[arg(long = "tape-length", aliases = ["tape_length", "tape", "total-length", "total_length"], default_value_t = 1usize << 21)]
    tape_length: usize,

    #[arg(long = "status", aliases = ["status-interval", "status_interval", "interval"], default_value_t = 10_000_000u64)]
    status_interval: u64,

    #[arg(long = "max-memory", aliases = ["max_memory", "max"], default_value_t = 1usize << 21)]
    max_total: usize,

    #[arg(long = "max-steps", aliases = ["max_steps"])]
    max_steps: Option<u64>,
}

fn main() {
    let args = Args::parse();
    let mut tape_length = args.tape_length;
    let status_interval = args.status_interval;
    let max_total = args.max_total;
    let max_steps = args.max_steps;
    assert!(
        tape_length >= 3,
        "tape_length must be >= 3 (two sentinels + at least one interior)"
    );
    if tape_length > max_total {
        eprintln!(
            "requested tape_length {} exceeds max {}, clamping",
            tape_length.separate_with_commas(),
            max_total.separate_with_commas()
        );
        tape_length = max_total;
    }
    let mut next_report: u64 = if status_interval == 0 {
        u64::MAX
    } else {
        status_interval
    };

    // tape with sentinel cells (value 2) at both ends
    // Interpret `tape_length` as TOTAL length (including the two sentinel cells).
    // (checked above) tape_length >= 3
    let mut tape: Vec<u8> = vec![0; tape_length];
    tape[0] = 2;
    tape[tape_length - 1] = 2;

    // Center of the interior [1 .. tape_length-1)
    let mut head_pointer = unsafe { tape.as_mut_ptr().add(1 + ((tape_length - 2) >> 1)) };
    let mut step_count: u64 = 0;
    let mut state_id: u8 = 0; // 0=A, 1=B, 2=C, 3=D, 4=E

    loop {
        // Choose heartbeat for this chunk: only yield when we have a reason
        // Choose heartbeat for this chunk: yield at the earliest of
        // - next report threshold (if enabled)
        // - max steps cap (if provided)
        // - otherwise, run until boundary/halt
        let hb_to_limit = max_steps
            .map(|limit| limit.saturating_sub(step_count))
            .unwrap_or(u64::MAX);
        let hb_to_report = if status_interval > 0 {
            next_report.saturating_sub(step_count)
        } else {
            u64::MAX
        };
        let hb_this_chunk: u64 = hb_to_limit.min(hb_to_report).max(1);

        let (new_head, status_code, steps_taken_this_chunk, new_state_id) =
            unsafe { bb6_contender_heartbeat(head_pointer, state_id, hb_this_chunk) };
        head_pointer = new_head;
        step_count += steps_taken_this_chunk;
        state_id = new_state_id;

        if let Some(limit) = max_steps {
            if step_count >= limit {
                println!(
                    "reached max steps {}; stopping",
                    step_count.separate_with_commas()
                );
                break;
            }
        }

        if status_code == 0 {
            if step_count >= next_report {
                // Print the greatest multiple of status_interval not exceeding step_count
                let crossed = (step_count - next_report) / status_interval + 1;
                let last = next_report + (crossed - 1) * status_interval;
                println!("{} steps", last.separate_with_commas());
                next_report = last.saturating_add(status_interval);
            }
            continue;
        }

        if status_code == 1 {
            println!("halted after {} steps", step_count.separate_with_commas());
            break;
        }

        // status_code == 2 => boundary reached; reallocate and continue
        let left_sentinel_ptr = tape.as_ptr();
        let right_sentinel_ptr = unsafe { tape.as_ptr().add(tape_length - 1) };
        let hit_left = head_pointer.cast_const() == left_sentinel_ptr;
        let hit_right = head_pointer.cast_const() == right_sentinel_ptr;

        if hit_left {
            match extend_tape_left(&mut tape, &mut tape_length, max_total) {
                Some(ptr) => head_pointer = ptr,
                None => {
                    println!(
                        "reached max memory {} cells; stopping at {} steps",
                        (max_total - 2).separate_with_commas(),
                        step_count.separate_with_commas()
                    );
                    break;
                }
            }
        } else if hit_right {
            match extend_tape_right(&mut tape, &mut tape_length, max_total) {
                Some(ptr) => head_pointer = ptr,
                None => {
                    println!(
                        "reached max memory {} cells; stopping at {} steps",
                        (max_total - 2).separate_with_commas(),
                        step_count.separate_with_commas()
                    );
                    break;
                }
            }
        } else {
            panic!("unexpected boundary pointer returned from asm");
        }

        if step_count >= next_report {
            let crossed = (step_count - next_report) / status_interval + 1;
            let last = next_report + (crossed - 1) * status_interval;
            println!("{} steps", last.separate_with_commas());
            next_report = last.saturating_add(status_interval);
        }
    }
}

/// Executes up to HEARTBEAT steps starting at `head`.
/// Returns (new_head, status_code, remaining_steps)
/// - `status_code`: 0 = ran HEARTBEAT, 1 = halted, 2 = boundary encountered
/// - remaining_steps: RCX after exit; steps_taken = HEARTBEAT - remaining_steps
unsafe fn bb5_champ_heartbeat(
    mut head: *mut u8,
    mut state_id: u8,
    heartbeat: u64,
) -> (*mut u8, u8, u64, u8) {
    let mut status_code: u8;
    let steps_taken: u64;
    unsafe {
        core::arch::asm!(
            tm_prog!(
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
unsafe fn bb6_contender_heartbeat(
    mut head: *mut u8,
    mut state_id: u8,
    heartbeat: u64,
) -> (*mut u8, u8, u64, u8) {
    let mut status_code: u8;
    let steps_taken: u64;
    unsafe {
        core::arch::asm!(
            tm_prog!(
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
