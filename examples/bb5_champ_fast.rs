#![allow(named_asm_labels)] // Allow alphabetic labels in inline asm for readability
use clap::Parser;
use thousands::Separable;

fn parse_usize_like_str(input: &str) -> Result<usize, String> {
    let s = input.replace('_', "");
    if let Ok(v) = s.parse::<usize>() {
        return Ok(v);
    }
    if let Some(exp) = s.strip_prefix("2^") {
        return exp
            .parse::<u32>()
            .map(|p| 2usize.saturating_pow(p))
            .map_err(|e| e.to_string());
    }
    if let Some(exp) = s.strip_prefix("1<<") {
        return exp
            .parse::<u32>()
            .map(|shift| 1usize.checked_shl(shift).unwrap_or(0))
            .map_err(|e| e.to_string());
    }
    Err(format!("invalid usize: {}", input))
}

fn parse_u64_like_str(input: &str) -> Result<u64, String> {
    let s = input.replace('_', "");
    if let Ok(v) = s.parse::<u64>() {
        return Ok(v);
    }
    if let Some(exp) = s.strip_prefix("2^") {
        return exp
            .parse::<u32>()
            .map(|p| 2u64.saturating_pow(p))
            .map_err(|e| e.to_string());
    }
    if let Some(exp) = s.strip_prefix("1<<") {
        return exp
            .parse::<u32>()
            .map(|shift| 1u64.checked_shl(shift).unwrap_or(0))
            .map_err(|e| e.to_string());
    }
    Err(format!("invalid u64: {}", input))
}

fn compute_heartbeat(status_interval: u64, max_steps: Option<u64>) -> u64 {
    let default_hb: u64 = 10_000;
    if let Some(ms) = max_steps {
        // Aim for ~2000 chunks, clamp to [1, default_hb]
        let target = (ms / 2000).clamp(1, default_hb);
        if status_interval > 0 {
            return target.min(status_interval.max(1));
        }
        return target.max(1);
    }
    if status_interval > 0 {
        // Keep chunk smaller than interval to get frequent updates
        return (status_interval / 10).clamp(1, default_hb);
    }
    default_hb
}
#[derive(Debug, Parser, Clone)]
#[command(name = "bb5_champ_fast", about = "Fast BB5 runner with inline asm")]
struct Args {
    #[arg(long = "tape-length", aliases = ["tape_length", "tape", "total-length", "total_length"], value_parser = parse_usize_like_str, default_value_t = (1usize << 21))]
    tape_length: usize,

    #[arg(long = "status", aliases = ["status-interval", "status_interval", "interval"], value_parser = parse_u64_like_str, default_value_t = 1_000u64)]
    status_interval: u64,

    #[arg(long = "heartbeat", aliases = ["hb"], value_parser = parse_u64_like_str)]
    heartbeat: Option<u64>,

    #[arg(long = "max-memory", aliases = ["max_memory", "max"], value_parser = parse_usize_like_str, default_value_t = (1usize << 21))]
    max_total: usize,

    #[arg(long = "max-steps", aliases = ["max_steps"], value_parser = parse_u64_like_str)]
    max_steps: Option<u64>,
}

fn main() {
    let args = Args::parse();
    let mut tape_length = args.tape_length;
    let status_interval = args.status_interval;
    let max_total = args.max_total;
    let max_steps = args.max_steps;
    let heartbeat: u64 = args
        .heartbeat
        .unwrap_or_else(|| compute_heartbeat(status_interval, max_steps));
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
        let (new_head, status_code, steps_taken_this_chunk, new_state_id) =
            unsafe { bb5_champ_heartbeat(head_pointer, state_id, heartbeat) };
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
#[allow(clippy::too_many_lines)]
unsafe fn bb5_champ_heartbeat(
    mut head: *mut u8,
    mut state_id: u8,
    heartbeat: u64,
) -> (*mut u8, u8, u64, u8) {
    let mut status_code: u8;
    let steps_taken: u64;
    unsafe {
        core::arch::asm!(
            "xor eax, eax",                  // status_code = 0 (AL)
            "mov rcx, {hb}",                 // loop counter: heartbeat steps
            // Dispatch to current state based on BL (state_id)
            "cmp bl, 0",                       // dispatch: BL == 0 (state A)?
            "je A",                            // if A, jump to state A
            "cmp bl, 1",                       // dispatch: BL == 1 (state B)?
            "je B",                            // if B, jump to state B
            "cmp bl, 2",                       // dispatch: BL == 2 (state C)?
            "je C",                            // if C, jump to state C
            "cmp bl, 3",                       // dispatch: BL == 3 (state D)?
            "je D",                            // if D, jump to state D
            "cmp bl, 4",                       // dispatch: BL == 4 (state E)?
            "je E",                            // if E, jump to state E
            "jmp A",                           // default to state A
            // State A
            "A:",                             // label: state A
            "cmp rcx, 0",                     // any steps remaining?
            "jne A_CONT",                     // continue unless heartbeat exhausted
            "mov bl, 0",                      // out of credit: resume at A
            "jmp END",                        // exit
            "A_CONT:",                        // continue A
            "mov dl, [rsi]",                  // load tape cell into DL
            "cmp dl, 2",                      // boundary sentinel check
            "je BOUNDARY_A",                   // rare: boundary in A
            "test dl, dl",                    // cell == 0 ?
            "jnz A_ONE",                      // if 1, branch A(1)
            "mov byte ptr [rsi], 1",          // A(0): write 1
            "inc rsi",                        // A(0): move Right
            "sub rcx, 1",                     // consume one step (post-decrement)
            "jmp B",                           // A(0): goto state B
            "A_ONE:",                          // label: A(1)
            "mov byte ptr [rsi], 1",          // A(1): write 1
            "dec rsi",                        // A(1): move Left
            "sub rcx, 1",                     // consume one step (post-decrement)
            "jmp C",                           // A(1): goto state C
            "BOUNDARY_A:",                    // boundary while in A
            "mov bl, 0",                      // record state A
            "jmp BOUNDARY",                   // handle boundary
            // State B
            "B:",                             // label: state B
            "cmp rcx, 0",                     // any steps remaining?
            "jne B_CONT",                     // continue unless heartbeat exhausted
            "mov bl, 1",                      // out of credit: resume at B
            "jmp END",                        // exit
            "B_CONT:",                        // continue B
            "mov dl, [rsi]",                  // load tape cell
            "cmp dl, 2",                      // boundary sentinel?
            "je BOUNDARY_B",                   // rare: boundary in B
            "test dl, dl",                    // cell == 0 ?
            "jnz B_ONE",                      // if 1, branch B(1)
            "mov byte ptr [rsi], 1",          // B(0): write 1
            "inc rsi",                        // B(0): move Right
            "sub rcx, 1",                     // consume one step
            "jmp C",                           // B(0): goto state C
            "B_ONE:",                          // label: B(1)
            "mov byte ptr [rsi], 1",          // B(1): write 1
            "inc rsi",                        // B(1): move Right
            "sub rcx, 1",                     // consume one step
            "jmp B",                           // B(1): goto state B (back)
            "BOUNDARY_B:",                    // boundary while in B
            "mov bl, 1",                      // record state B
            "jmp BOUNDARY",                   // handle boundary
            // State C
            "C:",                             // label: state C
            "cmp rcx, 0",                     // any steps remaining?
            "jne C_CONT",                     // continue unless heartbeat exhausted
            "mov bl, 2",                      // out of credit: resume at C
            "jmp END",                        // exit
            "C_CONT:",                        // continue C
            "mov dl, [rsi]",                  // load tape cell
            "cmp dl, 2",                      // boundary sentinel?
            "je BOUNDARY_C",                   // rare: boundary in C
            "test dl, dl",                    // cell == 0 ?
            "jnz C_ONE",                      // if 1, branch C(1)
            "mov byte ptr [rsi], 1",          // C(0): write 1
            "inc rsi",                        // C(0): move Right
            "sub rcx, 1",                     // consume one step
            "jmp D",                           // C(0): goto state D
            "C_ONE:",                          // label: C(1)
            "mov byte ptr [rsi], 0",          // C(1): write 0
            "dec rsi",                        // C(1): move Left
            "sub rcx, 1",                     // consume one step
            "jmp E",                           // C(1): goto state E
            "BOUNDARY_C:",                    // boundary while in C
            "mov bl, 2",                      // record state C
            "jmp BOUNDARY",                   // handle boundary
            // State D
            "D:",                             // label: state D
            "cmp rcx, 0",                     // any steps remaining?
            "jne D_CONT",                     // continue unless heartbeat exhausted
            "mov bl, 3",                      // out of credit: resume at D
            "jmp END",                        // exit
            "D_CONT:",                        // continue D
            "mov dl, [rsi]",                  // load tape cell
            "cmp dl, 2",                      // boundary sentinel?
            "je BOUNDARY_D",                   // rare: boundary in D
            "test dl, dl",                    // cell == 0 ?
            "jnz D_ONE",                      // if 1, branch D(1)
            "mov byte ptr [rsi], 1",          // D(0): write 1
            "dec rsi",                        // D(0): move Left
            "sub rcx, 1",                     // consume one step
            "jmp A",                           // D(0): goto state A
            "D_ONE:",                          // label: D(1)
            "mov byte ptr [rsi], 1",          // D(1): write 1
            "dec rsi",                        // D(1): move Left
            "sub rcx, 1",                     // consume one step
            "jmp D",                           // D(1): goto state D
            "BOUNDARY_D:",                    // boundary while in D
            "mov bl, 3",                      // record state D
            "jmp BOUNDARY",                   // handle boundary
            // State E
            "E:",                             // label: state E
            "cmp rcx, 0",                     // any steps remaining?
            "jne E_CONT",                     // continue unless heartbeat exhausted
            "mov bl, 4",                      // out of credit: resume at E
            "jmp END",                        // exit
            "E_CONT:",                        // continue E
            "mov dl, [rsi]",                  // load tape cell
            "cmp dl, 2",                      // boundary sentinel?
            "je BOUNDARY_E",                   // rare: boundary in E
            "test dl, dl",                    // cell == 0 ?
            "jnz E_ONE",                      // if 1, branch E(1)
            "mov byte ptr [rsi], 1",          // E(0): write 1
            "inc rsi",                        // E(0): move Right
            "mov al, 1",                      // E(0): set status_code = 1 (halt)
            "sub rcx, 1",                     // consume one step (post-decrement)
            "mov bl, 4",                      // halting from state E
            "jmp END",                         // E(0): end (halt)
            "E_ONE:",                          // label: E(1)
            "mov byte ptr [rsi], 0",          // E(1): write 0
            "dec rsi",                        // E(1): move Left
            "sub rcx, 1",                     // consume one step
            "jmp A",                           // E(1): goto state A
            "BOUNDARY_E:",                    // boundary while in E
            "mov bl, 4",                      // record state E
            "jmp BOUNDARY",                   // handle boundary
            // Boundary sentinel
            "BOUNDARY:",                      // label: boundary sentinel
            "mov al, 2",                      // status_code = 2 (boundary)
            "inc rcx",                        // undo loop counter decrement
            // End
            "END:",                           // label: end
            // steps_taken = heartbeat - remaining (rcx)
            "mov r8, {hb}",                  // R8 := heartbeat
            "sub r8, rcx",                   // steps_taken = heartbeat - remaining
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
