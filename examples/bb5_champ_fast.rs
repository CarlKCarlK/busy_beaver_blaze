#![allow(named_asm_labels)] // Allow alphabetic labels in inline asm for readability
use std::env;
use thousands::Separable;

fn main() {
    let mut tape_length: usize = env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(1 << 21); // about 2 Million cells
    assert!(
        tape_length >= 3,
        "tape_length must be >= 3 (two sentinels + at least one interior)"
    );
    let status_interval: u64 = env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(1_000);
    // Heartbeat (steps per asm chunk); allow CLI override, default 10_000
    let heartbeat: u64 = env::args()
        .nth(3)
        .and_then(|value| value.parse().ok())
        .unwrap_or(10_000);
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
            head_pointer = extend_tape_left(&mut tape, &mut tape_length);
        } else if hit_right {
            head_pointer = extend_tape_right(&mut tape, &mut tape_length);
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
            "dec rcx",                        // consume one step
            "jnz A_CONT",                     // continue unless heartbeat exhausted
            "mov bl, 0",                      // exiting from state A
            "jmp END",                        // end
            "A_CONT:",                        // continue A
            "mov dl, [rsi]",                  // load tape cell into DL
            "cmp dl, 2",                      // boundary sentinel check
            "jne A_NONBOUND",                 // normal path if not boundary
            "mov bl, 0",                      // boundary while in state A
            "jmp BOUNDARY",                   // handle boundary
            "A_NONBOUND:",                    // continue A (non-boundary)
            "test dl, dl",                    // cell == 0 ?
            "jnz A_ONE",                      // if 1, branch A(1)
            "mov byte ptr [rsi], 1",          // A(0): write 1
            "inc rsi",                        // A(0): move Right
            "jmp B",                           // A(0): goto state B
            "A_ONE:",                          // label: A(1)
            "mov byte ptr [rsi], 1",          // A(1): write 1
            "dec rsi",                        // A(1): move Left
            "jmp C",                           // A(1): goto state C
            // State B
            "B:",                             // label: state B
            "dec rcx",                        // consume one step
            "jnz B_CONT",                     // continue B unless heartbeat exhausted
            "mov bl, 1",                      // exiting from state B
            "jmp END",                        // exit heartbeat (resume in B)
            "B_CONT:",                        // continue B
            "mov dl, [rsi]",                  // load tape cell
            "cmp dl, 2",                      // boundary sentinel?
            "jne B_NONBOUND",                 // not boundary: continue B
            "mov bl, 1",                      // boundary while in state B
            "jmp BOUNDARY",                   // handle boundary
            "B_NONBOUND:",                    // continue B (non-boundary)
            "test dl, dl",                    // cell == 0 ?
            "jnz B_ONE",                      // if 1, branch B(1)
            "mov byte ptr [rsi], 1",          // B(0): write 1
            "inc rsi",                        // B(0): move Right
            "jmp C",                           // B(0): goto state C
            "B_ONE:",                          // label: B(1)
            "mov byte ptr [rsi], 1",          // B(1): write 1
            "inc rsi",                        // B(1): move Right
            "jmp B",                           // B(1): goto state B (back)
            // State C
            "C:",                             // label: state C
            "dec rcx",                        // consume one step
            "jnz C_CONT",                     // continue C unless heartbeat exhausted
            "mov bl, 2",                      // exiting from state C
            "jmp END",                        // exit heartbeat (resume in C)
            "C_CONT:",                        // continue C
            "mov dl, [rsi]",                  // load tape cell
            "cmp dl, 2",                      // boundary sentinel?
            "jne C_NONBOUND",                 // not boundary: continue C
            "mov bl, 2",                      // boundary while in state C
            "jmp BOUNDARY",
            "C_NONBOUND:",                    // continue C (non-boundary)
            "test dl, dl",                    // cell == 0 ?
            "jnz C_ONE",                      // if 1, branch C(1)
            "mov byte ptr [rsi], 1",          // C(0): write 1
            "inc rsi",                        // C(0): move Right
            "jmp D",                           // C(0): goto state D
            "C_ONE:",                          // label: C(1)
            "mov byte ptr [rsi], 0",          // C(1): write 0
            "dec rsi",                        // C(1): move Left
            "jmp E",                           // C(1): goto state E
            // State D
            "D:",                             // label: state D
            "dec rcx",                        // consume one step
            "jnz D_CONT",                     // continue D unless heartbeat exhausted
            "mov bl, 3",                      // exiting from state D
            "jmp END",                        // exit heartbeat (resume in D)
            "D_CONT:",                        // continue D
            "mov dl, [rsi]",                  // load tape cell
            "cmp dl, 2",                      // boundary sentinel?
            "jne D_NONBOUND",                 // not boundary: continue D
            "mov bl, 3",                      // boundary while in state D
            "jmp BOUNDARY",                   // handle boundary
            "D_NONBOUND:",                    // continue D (non-boundary)
            "test dl, dl",                    // cell == 0 ?
            "jnz D_ONE",                      // if 1, branch D(1)
            "mov byte ptr [rsi], 1",          // D(0): write 1
            "dec rsi",                        // D(0): move Left
            "jmp A",                           // D(0): goto state A
            "D_ONE:",                          // label: D(1)
            "mov byte ptr [rsi], 1",          // D(1): write 1
            "dec rsi",                        // D(1): move Left
            "jmp D",                           // D(1): goto state D
            // State E
            "E:",                             // label: state E
            "dec rcx",                        // consume one step
            "jnz E_CONT",                     // continue E unless heartbeat exhausted
            "mov bl, 4",                      // exiting from state E
            "jmp END",                        // exit heartbeat (resume in E)
            "E_CONT:",                        // continue E
            "mov dl, [rsi]",                  // load tape cell
            "cmp dl, 2",                      // boundary sentinel?
            "jne E_NONBOUND",                 // not boundary: continue E
            "mov bl, 4",                      // boundary while in state E
            "jmp BOUNDARY",                   // handle boundary
            "E_NONBOUND:",                    // continue E (non-boundary)
            "test dl, dl",                    // cell == 0 ?
            "jnz E_ONE",                      // if 1, branch E(1)
            "mov byte ptr [rsi], 1",          // E(0): write 1
            "inc rsi",                        // E(0): move Right
            "mov al, 1",                      // E(0): set status_code = 1 (halt)
            "mov bl, 4",                      // halting from state E
            "jmp END",                         // E(0): end (halt)
            "E_ONE:",                          // label: E(1)
            "mov byte ptr [rsi], 0",          // E(1): write 0
            "dec rsi",                        // E(1): move Left
            "jmp A",                           // E(1): goto state A
            // Boundary sentinel
            "BOUNDARY:",                      // label: boundary sentinel
            "mov al, 2",                      // status_code = 2 (boundary)
            "inc rcx",                        // undo loop counter decrement
            // End
            "END:",                           // label: end
            // steps_taken = heartbeat - remaining (rcx)
            // Adjust for pre-decrement style: on full heartbeat exit (AL==0),
            // we did not execute the last step that decremented RCX to 0.
            "mov r8, {hb}",                  // R8 := heartbeat
            "sub r8, rcx",                   // steps_taken = HEARTBEAT - remaining
            "test al, al",                   // AL == 0? (full heartbeat)
            "jnz ADJUST_DONE",               // if non-zero (halt/boundary), skip adjust
            "dec r8",                        // adjust for pre-decrement on exact boundary
            "ADJUST_DONE:",                   // label: done adjusting steps
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

fn extend_tape_left(tape: &mut Vec<u8>, total_len: &mut usize) -> *mut u8 {
    let old_total = *total_len;
    let old_interior = old_total.saturating_sub(2);
    let growth = old_interior.max(1);
    let new_total = old_total + growth;
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
    unsafe { tape.as_mut_ptr().add(growth) }
}

fn extend_tape_right(tape: &mut Vec<u8>, total_len: &mut usize) -> *mut u8 {
    let old_total = *total_len;
    let old_interior = old_total.saturating_sub(2);
    let growth = old_interior.max(1);
    let new_total = old_total + growth;
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
    unsafe { tape.as_mut_ptr().add(old_total - 1) }
}
