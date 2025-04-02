import itertools
from busy_beaver_blaze import Machine


def test_machine_halting():
    """Test that a specific machine halts after the expected number of steps with the correct number of ones."""
    machine = Machine("1RB1LC_0LA0LD_1LA1RZ_1LB1RE_0RD0RB")
    step_limit = 10_000_000

    # Count steps until halting or limit
    steps_run = 1 + sum(1 for _ in itertools.islice(machine, step_limit))
    ones_count = machine.count_ones()
    is_halted = machine.is_halted()

    # Check results
    expected_steps = 2_133_492
    expected_ones = 1_915
    assert is_halted, "Machine did not halt as expected"

    assert steps_run == expected_steps, (
        f"Expected {expected_steps} steps, got {steps_run}"
    )
    assert ones_count == expected_ones, (
        f"Expected {expected_ones} ones, got {ones_count}"
    )

    print(
        f"Test passed: Machine halted after {steps_run:,} steps with {ones_count:,} ones on the tape"
    )


def test_machine_non_halting():
    """Test a machine that doesn't halt within a step limit but has the expected number of ones."""
    import itertools

    # Multi-line format as specified in the comment
    # The standard format is a one-line representation with states separated by underscores
    # Each state has two actions (for 0 and 1) in the format: [write][direction][next_state]
    program_text = "1RB0RE_0RC0RA_1LD---_1LA0LB_1RA0LC"

    machine = Machine(program_text)
    step_limit = 100_000

    # Run for up to the step limit
    steps_run = 1 + sum(1 for _ in itertools.islice(machine, step_limit - 1))
    ones_count = machine.count_ones()
    is_halted = machine.is_halted()

    # Check results
    expected_steps = step_limit  # Should reach the limit without halting
    expected_ones = 49

    assert not is_halted, "Machine unexpectedly halted"

    assert steps_run == expected_steps, (
        f"Expected {expected_steps} steps, got {steps_run}"
    )
    assert ones_count == expected_ones, (
        f"Expected {expected_ones} ones, got {ones_count}"
    )
    assert not machine.is_halted(), "Machine unexpectedly halted"

    print(
        f"Test passed: Machine ran for {steps_run:,} steps without halting, with {ones_count} ones on the tape"
    )


def test_machine_transposed_format():
    """Test a machine defined in the transposed format (symbol-to-state)."""
    import itertools

    # Transposed format (symbol-to-state)
    program_text = "1RB1RE_1LC1RD_1LA0LC_1RA---_1LE0RB"

    machine = Machine(program_text)
    step_limit = 10

    # Run for exactly 10 steps
    steps_run = 1 + sum(1 for _ in itertools.islice(machine, step_limit - 1))
    ones_count = machine.count_ones()
    is_halted = machine.is_halted()

    # Check results
    expected_steps = step_limit
    expected_ones = 5

    assert not is_halted, "Machine unexpectedly halted"
    assert steps_run == expected_steps, (
        f"Expected {expected_steps} steps, got {steps_run}"
    )
    assert ones_count == expected_ones, (
        f"Expected {expected_ones} ones, got {ones_count}"
    )

    print(
        f"Test passed: Machine ran for {steps_run} steps without halting, with {ones_count} ones on the tape"
    )
