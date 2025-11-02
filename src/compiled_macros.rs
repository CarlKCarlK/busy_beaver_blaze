//! Macros for generating inline assembly Turing machine steppers.
//!
//! These macros are shared between the library (`compiled_runner.rs`) and
//! the standalone example (`examples/compile_machine.rs`).

#![allow(unused_macros)]

/// Generate assembly instruction to move the tape head.
/// - `R`: move right (add rsi, 1)
/// - `L`: move left (sub rsi, 1)
#[macro_export]
macro_rules! tm_move {
    (R) => {
        "add rsi, 1\n"
    };
    (L) => {
        "sub rsi, 1\n"
    };
}

/// Helper to create a single line of assembly with trailing newline.
#[macro_export]
macro_rules! asmline {
    ($($p:expr),* $(,)?) => {
        concat!($($p),*, "\n")
    };
}

/// Stringify helper for use in assembly templates.
#[macro_export]
macro_rules! s {
    ($x:expr) => {
        stringify!($x)
    };
}

/// Generate conditional store instruction on 0-branch (when reading 0).
/// Optimizes out redundant writes: writing 0 on 0-branch is a no-op.
#[macro_export]
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

/// Generate conditional store instruction on 1-branch (when reading 1).
/// Optimizes out redundant writes: writing 1 on 1-branch is a no-op.
#[macro_export]
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

/// Generate transition to next state after processing current cell.
/// Handles both HALT and regular state transitions.
#[macro_export]
macro_rules! tm_next {
    ($P:ident, HALT, $id:expr) => {
        concat!(
            asmline!("dec r10"),
            asmline!("mov al, 1"),
            asmline!("mov bl, ", s!($id)),
            asmline!("jmp ", s!($P), "_END"),
        )
    };
    ($P:ident, $N:ident, $id:expr) => {
        concat!(
            asmline!("dec r10"),
            asmline!("jz ", s!($P), "_END"),
            asmline!("jmp ", s!($P), "_", s!($N)),
        )
    };
}

/// Generate dispatch table entry comparing current state with target state.
#[macro_export]
macro_rules! tm_dispatch {
    ( $P:ident, $S:ident, $id:expr ) => {
        concat!(
            asmline!("cmp bl, ", s!($id)),
            asmline!("je ", s!($P), "_", s!($S)),
        )
    };
}

/// Generate complete state block handling 0/1 read branches.
#[macro_export]
macro_rules! tm_state_block {
    ( $P:ident, $S:ident, $id:expr, ( $w0:literal, $d0:ident, $n0:ident ), ( $w1:literal, $d1:ident, $n1:ident ) ) => {
        concat!(
            asmline!(s!($P), "_", s!($S), ":"),
            asmline!("mov bl, ", s!($id)),
            asmline!("mov dl, [rsi]"),
            asmline!("cmp dl, 255"),
            asmline!("je ", s!($P), "_BOUNDARY_", s!($S)),
            asmline!("test dl, dl"),
            asmline!("jnz ", s!($P), "_", s!($S), "_ONE"),
            tm_store_on_0!($w0),
            tm_move!($d0),
            tm_next!($P, $n0, $id),
            asmline!(s!($P), "_", s!($S), "_ONE:"),
            tm_store_on_1!($w1),
            tm_move!($d1),
            tm_next!($P, $n1, $id),
            asmline!(s!($P), "_BOUNDARY_", s!($S), ":"),
            asmline!("mov bl, ", s!($id)),
            asmline!("jmp ", s!($P), "_BOUNDARY"),
        )
    };
}

/// Generate complete program template with dispatch and all states.
#[macro_export]
macro_rules! tm_prog {
    ( $P:ident, ($S0:ident, $id0:expr, $z0:tt, $o0:tt) $(, ($S:ident, $id:expr, $z:tt, $o:tt) )* $(,)? ) => {
        concat!(
            asmline!("xor eax, eax"),
            asmline!("mov r10, r9"),
            tm_dispatch!($P, $S0, $id0),
            $( tm_dispatch!($P, $S, $id), )*
            asmline!("jmp ", s!($P), "_", s!($S0)),
            tm_state_block!($P, $S0, $id0, $z0, $o0),
            $( tm_state_block!($P, $S, $id, $z, $o), )*
            asmline!(s!($P), "_BOUNDARY:"),
            asmline!("mov al, 2"),
            asmline!(s!($P), "_END:"),
            asmline!("mov r8, r9"),
            asmline!("sub r8, r10"),
        )
    };
}
