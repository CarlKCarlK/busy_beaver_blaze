use std::env;

const HEARTBEAT: usize = 10_000;

fn main() {
    let tape_length: usize = env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(1 << 21);
    let status_interval: u64 = env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(1_000_000);

    // tape with sentinel cells (value 2) at both ends
    let mut tape: Vec<u8> = vec![0; tape_length + 2];
    tape[0] = 2;
    tape[tape_length + 1] = 2;
    let mut head_pointer = unsafe { tape.as_mut_ptr().add(1 + tape_length / 2) };

    let mut step_count: u64 = 1;
    loop {
        let (new_head, halted) = unsafe { bb5_champ_heartbeat(head_pointer) };
        head_pointer = new_head;
        step_count += HEARTBEAT as u64;

        if step_count % status_interval == 0 {
            println!("{step_count} steps");
        }

        if halted {
            println!("halted after {step_count} steps");
            break;
        }
    }
}

extern "C" fn boundary_handler() -> ! {
    panic!("tape boundary reached");
}

/// Executes HEARTBEAT steps of the BB5 champion Turing machine starting at `head`.
///
/// # Safety
/// `head` must point into a tape with at least HEARTBEAT cells available on both sides.
unsafe fn bb5_champ_heartbeat(mut head: *mut u8) -> (*mut u8, bool) {
    let halted: u8;
    unsafe {
        core::arch::asm!(
            "xor eax, eax",
            "mov rcx, {hb}",
            "jmp 2f",
            // State A
            "2:",
            "dec rcx",
            "jz 22f",
            "mov dl, [rsi]",
            "cmp dl, 2",
            "je 30f",
            "test dl, dl",
            "jnz 3f",
            "mov byte ptr [rsi], 1",
            "inc rsi",
            "jmp 4f",
            "3:",
            "mov byte ptr [rsi], 1",
            "dec rsi",
            "jmp 6f",
            // State B
            "4:",
            "dec rcx",
            "jz 22f",
            "mov dl, [rsi]",
            "cmp dl, 2",
            "je 30f",
            "test dl, dl",
            "jnz 5f",
            "mov byte ptr [rsi], 1",
            "inc rsi",
            "jmp 6f",
            "5:",
            "mov byte ptr [rsi], 1",
            "inc rsi",
            "jmp 4b",
            // State C
            "6:",
            "dec rcx",
            "jz 22f",
            "mov dl, [rsi]",
            "cmp dl, 2",
            "je 30f",
            "test dl, dl",
            "jnz 7f",
            "mov byte ptr [rsi], 1",
            "inc rsi",
            "jmp 8f",
            "7:",
            "mov byte ptr [rsi], 0",
            "dec rsi",
            "jmp 20f",
            // State D
            "8:",
            "dec rcx",
            "jz 22f",
            "mov dl, [rsi]",
            "cmp dl, 2",
            "je 30f",
            "test dl, dl",
            "jnz 9f",
            "mov byte ptr [rsi], 1",
            "dec rsi",
            "jmp 2b",
            "9:",
            "mov byte ptr [rsi], 1",
            "dec rsi",
            "jmp 8b",
            // State E
            "20:",
            "dec rcx",
            "jz 22f",
            "mov dl, [rsi]",
            "cmp dl, 2",
            "je 30f",
            "test dl, dl",
            "jnz 21f",
            "mov byte ptr [rsi], 1",
            "inc rsi",
            "mov al, 1",
            "jmp 22f",
            "21:",
            "mov byte ptr [rsi], 0",
            "dec rsi",
            "jmp 2b",
            // Boundary sentinel
            "30:",
            "call {boundary}",
            "ud2",
            // End
            "22:",
            hb = const HEARTBEAT,
            inout("rsi") head,
            lateout("al") halted,
            boundary = sym boundary_handler,
            out("rcx") _,
            out("rdx") _,
            options(nostack)
        );
    }
    (head, halted != 0)
}

