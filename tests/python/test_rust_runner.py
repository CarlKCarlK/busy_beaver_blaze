"""Tests for the rust-backed run_machine_steps helper."""

import pytest

pytest.importorskip("busy_beaver_blaze._busy_beaver_blaze")

from busy_beaver_blaze import Machine, run_machine_steps

if run_machine_steps is None:
    pytest.skip("Rust extension is not available", allow_module_level=True)


def _run_python_machine(program_text: str, step_limit: int) -> tuple[int, int]:
    """Simulate the pure Python machine up to step_limit."""
    machine = Machine(program_text)
    steps_run = 0

    for _ in range(step_limit):
        try:
            next(machine)
        except StopIteration:
            steps_run += 1
            break
        steps_run += 1
    else:
        return steps_run, machine.count_ones()

    nonzero_count = machine.count_ones()
    return steps_run, nonzero_count


def test_run_machine_steps_matches_python() -> None:
    program_text = "1RB1LA_1LB1RH"
    step_limit = 10

    python_steps, python_nonzeros = _run_python_machine(program_text, step_limit)
    rust_steps, rust_nonzeros = run_machine_steps(program_text, step_limit)

    assert rust_steps == python_steps
    assert rust_nonzeros == python_nonzeros


def test_run_machine_steps_respects_limit() -> None:
    program_text = "1RB1LA_1LB1RH"
    step_limit = 1

    python_steps, python_nonzeros = _run_python_machine(program_text, step_limit)
    rust_steps, rust_nonzeros = run_machine_steps(program_text, step_limit)

    assert rust_steps == python_steps == 1
    assert rust_nonzeros == python_nonzeros == 1


def test_step_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="step_limit must be at least 1"):
        run_machine_steps("1RB1LA_1LB1RH", 0)


def test_force_python_only_flag() -> None:
    program_text = "1RB1LA_1LB1RH"
    step_limit = 10

    rust_steps, rust_nonzeros = run_machine_steps(program_text, step_limit)
    python_steps, python_nonzeros = run_machine_steps(
        program_text,
        step_limit,
        force_python_only=True,
    )

    assert python_steps == rust_steps
    assert python_nonzeros == rust_nonzeros
