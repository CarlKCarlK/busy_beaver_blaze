"""Tests for the rust-backed run_machine_steps helper."""

import pytest

pytest.importorskip("busy_beaver_blaze._busy_beaver_blaze")

from busy_beaver_blaze import BB5_CHAMP, Machine, run_machine_steps


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


def test_force_modes() -> None:
    program_text = "1RB1LA_1LB1RH"
    step_limit = 10

    rust_steps, rust_nonzeros = run_machine_steps(program_text, step_limit, force="rust")
    python_steps, python_nonzeros = run_machine_steps(program_text, step_limit, force="python")

    assert python_steps == rust_steps
    assert python_nonzeros == rust_nonzeros

    auto_steps, auto_nonzeros = run_machine_steps(program_text, step_limit)
    assert auto_steps == rust_steps
    assert auto_nonzeros == rust_nonzeros


def test_invalid_force_value() -> None:
    with pytest.raises(ValueError, match="force must be None"):
        run_machine_steps("1RB1LA_1LB1RH", 10, force="invalid")


def test_auto_selects_asm_for_bb5() -> None:
    manual_steps, manual_nonzeros = run_machine_steps(BB5_CHAMP, 1_000, force="rust")
    asm_steps, asm_nonzeros = run_machine_steps(BB5_CHAMP, 1_000)

    assert manual_steps == asm_steps
    assert manual_nonzeros == asm_nonzeros
