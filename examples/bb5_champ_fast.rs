use std::env;

const HEARTBEAT: usize = 10_000;

fn main() {
    let mut tape_length: usize = env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(1 << 21);
    let status_interval: u64 = env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(1_000);
    let mut next_report: u64 = if status_interval == 0 { u64::MAX } else { status_interval };

    // tape with sentinel cells (value 2) at both ends
    let mut tape: Vec<u8> = vec![0; tape_length + 2];
    tape[0] = 2;
    tape[tape_length + 1] = 2;

    let mut head_pointer = unsafe { tape.as_mut_ptr().add((1 + tape_length) >> 1) };
    let mut step_count: u64 = 0;

    loop {
        let (new_head, status_code, steps_taken_this_chunk) =
            unsafe { bb5_champ_heartbeat(head_pointer) };
        head_pointer = new_head;
        step_count += steps_taken_this_chunk;

        if status_code == 0 {
            if step_count >= next_report {
                println!("{step_count} steps");
                // advance to the next multiple; avoid overflow if interval huge
                next_report = next_report.saturating_add(status_interval);
            }
            continue;
        }

        if status_code == 1 {
            println!("halted after {step_count} steps");
            break;
        }

        // status_code == 2 => boundary reached; reallocate and continue
        let left_sentinel_ptr = tape.as_ptr();
        let right_sentinel_ptr = unsafe { tape.as_ptr().add(tape_length + 1) };
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
            println!("{step_count} steps");
            next_report = next_report.saturating_add(status_interval);
        }
    }
}

// BUG: really want it to re-allocate. (handled by extend_tape_left/right)

/// Executes up to HEARTBEAT steps starting at `head`.
/// Returns (new_head, status_code, remaining_steps)
/// - status_code: 0 = ran HEARTBEAT, 1 = halted, 2 = boundary encountered
/// - remaining_steps: RCX after exit; steps_taken = HEARTBEAT - remaining_steps
unsafe fn bb5_champ_heartbeat(mut head: *mut u8) -> (*mut u8, u8, u64) {
    let mut status_code: u8;
    let steps_taken: u64;
    unsafe {
        core::arch::asm!(
            "xor eax, eax",                  // status_code = 0 (AL)
            "mov rcx, {hb}",                 // loop counter: HEARTBEAT steps
            "xor r8d, r8d",                  // steps_taken = 0 (R8D)
            "jmp 2f",                        // jump to State A entry
            // State A
            "2:",                            // label: state A
            "dec rcx",                       // consume one step
            "inc r8d",                       // steps_taken += 1
            "jz 25f",                        // end if no steps left
            "mov dl, [rsi]",                 // load tape cell into DL
            "cmp dl, 2",                     // boundary sentinel check
            "je 24f",                        // if sentinel, handle boundary
            "test dl, dl",                   // cell == 0 ?
            "jnz 3f",                        // if 1, branch A(1)
            "mov byte ptr [rsi], 1",         // A(0): write 1
            "inc rsi",                       // A(0): move Right
            "jmp 4f",                        // A(0): goto state B
            "3:",                            // label: A(1)
            "mov byte ptr [rsi], 1",         // A(1): write 1
            "dec rsi",                       // A(1): move Left
            "jmp 6f",                        // A(1): goto state C
            // State B
            "4:",                            // label: state B
            "dec rcx",                       // consume one step
            "inc r8d",                       // steps_taken += 1
            "jz 25f",                        // end if no steps left
            "mov dl, [rsi]",                 // load tape cell
            "cmp dl, 2",                     // boundary sentinel?
            "je 24f",                        // handle boundary
            "test dl, dl",                   // cell == 0 ?
            "jnz 5f",                        // if 1, branch B(1)
            "mov byte ptr [rsi], 1",         // B(0): write 1
            "inc rsi",                       // B(0): move Right
            "jmp 6f",                        // B(0): goto state C
            "5:",                            // label: B(1)
            "mov byte ptr [rsi], 1",         // B(1): write 1
            "inc rsi",                       // B(1): move Right
            "jmp 4b",                        // B(1): goto state B (back)
            // State C
            "6:",                            // label: state C
            "dec rcx",                       // consume one step
            "inc r8d",                       // steps_taken += 1
            "jz 25f",                        // end if no steps left
            "mov dl, [rsi]",                 // load tape cell
            "cmp dl, 2",                     // boundary sentinel?
            "je 24f",                        // handle boundary
            "test dl, dl",                   // cell == 0 ?
            "jnz 7f",                        // if 1, branch C(1)
            "mov byte ptr [rsi], 1",         // C(0): write 1
            "inc rsi",                       // C(0): move Right
            "jmp 8f",                        // C(0): goto state D
            "7:",                            // label: C(1)
            "mov byte ptr [rsi], 0",         // C(1): write 0
            "dec rsi",                       // C(1): move Left
            "jmp 22f",                       // C(1): goto state E
            // State D
            "8:",                            // label: state D
            "dec rcx",                       // consume one step
            "inc r8d",                       // steps_taken += 1
            "jz 25f",                        // end if no steps left
            "mov dl, [rsi]",                 // load tape cell
            "cmp dl, 2",                     // boundary sentinel?
            "je 24f",                        // handle boundary
            "test dl, dl",                   // cell == 0 ?
            "jnz 9f",                        // if 1, branch D(1)
            "mov byte ptr [rsi], 1",         // D(0): write 1
            "dec rsi",                       // D(0): move Left
            "jmp 2b",                        // D(0): goto state A
            "9:",                            // label: D(1)
            "mov byte ptr [rsi], 1",         // D(1): write 1
            "dec rsi",                       // D(1): move Left
            "jmp 8b",                        // D(1): goto state D
            // State E
            "22:",                           // label: state E
            "dec rcx",                       // consume one step
            "inc r8d",                       // steps_taken += 1
            "jz 25f",                        // end if no steps left
            "mov dl, [rsi]",                 // load tape cell
            "cmp dl, 2",                     // boundary sentinel?
            "je 24f",                        // handle boundary
            "test dl, dl",                   // cell == 0 ?
            "jnz 23f",                       // if 1, branch E(1)
            "mov byte ptr [rsi], 1",         // E(0): write 1
            "inc rsi",                       // E(0): move Right
            "mov al, 1",                     // E(0): set status_code = 1 (halt)
            "jmp 25f",                       // E(0): end (halt)
            "23:",                           // label: E(1)
            "mov byte ptr [rsi], 0",         // E(1): write 0
            "dec rsi",                       // E(1): move Left
            "jmp 2b",                        // E(1): goto state A
            // Boundary sentinel
            "24:",                           // label: boundary sentinel
            "mov al, 2",                     // status_code = 2 (boundary)
            "inc rcx",                       // undo step consumption; retry after grow
            // End
            "25:",                           // label: end
            inout("rsi") head,                // head pointer in/out
            lateout("r8") steps_taken,        // steps taken this heartbeat
            lateout("al") status_code,        // status code in AL
            out("rdx") _,                     // clobber DL container
            out("rcx") _,                     // clobber: RCX used as loop counter
            hb = const HEARTBEAT,
            options(nostack)
        );
    };
    (head, status_code, steps_taken)
}

fn extend_tape_left(tape: &mut Vec<u8>, tape_length: &mut usize) -> *mut u8 {
    let old_length = *tape_length;
    let growth = old_length.max(1);
    let new_length = old_length + growth;
    let mut new_tape = vec![0u8; new_length + 2];
    new_tape[0] = 2;
    new_tape[new_length + 1] = 2;
    // Copy old interior shifted right by `growth`.
    let dst_start = 1 + growth;
    let dst_end = dst_start + old_length;
    new_tape[dst_start..dst_end].copy_from_slice(&tape[1..(1 + old_length)]);
    *tape = new_tape;
    *tape_length = new_length;
    // Head should be at the first newly-added interior cell.
    unsafe { tape.as_mut_ptr().add(growth) }
}

fn extend_tape_right(tape: &mut Vec<u8>, tape_length: &mut usize) -> *mut u8 {
    let old_length = *tape_length;
    let growth = old_length.max(1);
    let new_length = old_length + growth;
    let mut new_tape = vec![0u8; new_length + 2];
    new_tape[0] = 2;
    new_tape[new_length + 1] = 2;
    // Copy old interior at the same offset.
    new_tape[1..(1 + old_length)].copy_from_slice(&tape[1..(1 + old_length)]);
    *tape = new_tape;
    *tape_length = new_length;
    // Head should be the first newly-added interior cell to the right of the old end.
    unsafe { tape.as_mut_ptr().add(old_length + 1) }
}
