class Tape:
    def __init__(self):
        self.negative = []  # Cells at negative positions (reversed)
        self.nonnegative = [
            False
        ]  # Cells at non-negative positions, starting with position 0

    def __getitem__(self, position):
        if position >= 0:
            return (
                self.nonnegative[position]
                if position < len(self.nonnegative)
                else False
            )
        else:
            neg_index = -position - 1
            return self.negative[neg_index] if neg_index < len(self.negative) else False

    def __setitem__(self, position, value):
        if position >= 0:
            while position >= len(self.nonnegative):
                self.nonnegative.append(False)
            self.nonnegative[position] = value
        else:
            neg_index = -position - 1
            while neg_index >= len(self.negative):
                self.negative.append(False)
            self.negative[neg_index] = value

    def __str__(self):
        negative_part = "".join(["1" if x else "0" for x in reversed(self.negative)])
        nonnegative_part = "".join(["1" if x else "0" for x in self.nonnegative])
        return f"{negative_part}|{nonnegative_part}"

    def count_ones(self):
        """Count the number of 1s on the tape"""
        ones_count = 0
        for cell in self.negative:
            if cell:
                ones_count += 1
        for cell in self.nonnegative:
            if cell:
                ones_count += 1
        return ones_count


class Action:
    def __init__(self, next_state, next_symbol, direction):
        self.next_state = next_state
        self.next_symbol = next_symbol
        self.direction = direction


class Program:
    def __init__(self, state_count, actions):
        self.state_count = state_count
        self.actions = actions  # Flat list of actions indexed by state*2 + symbol

    def action(self, state, symbol):
        # Convert symbol from boolean to integer (0 or 1)
        symbol_int = 1 if symbol else 0
        # Calculate offset in flat actions list
        offset = state * 2 + symbol_int
        return self.actions[offset]

    @classmethod
    def from_text(cls, program_text):
        """Parse a program from text representation in one of three supported formats.

        Format 1 (state-to-symbol): Matrix format with states as rows and symbols as columns
        Example:
        ```
        0	1
        A	1RB	1RE
        B	1LC	0LC
        C	1RD	1LB
        D	1RA	0RD
        E	---	0RC
        ```

        Format 2 (symbol-to-state): Transposed matrix with symbols as rows and states as columns
        Example:
        ```
                A	B	C	D	E
        0	1RB	1LC	1RD	1RA	---
        1	1RE	0LC	1LB	0RD	0RC
        ```

        Format 3 (standard): Compact one-line format with states separated by underscores
        Example: "1RB1RE_1LC0LC_1RD1LB_1RA0RD_---0RC"
        """
        program_text = program_text.strip()

        # Determine which format is being used
        if "_" in program_text and program_text.count("\n") == 0:
            # Standard format (one line with underscores)
            return cls._parse_standard_format(program_text)
        elif program_text.split("\n")[0].strip().startswith(("0", "1")):
            # Symbol-to-state format (first non-empty line starts with 0 or 1)
            return cls._parse_symbol_to_state(program_text)
        else:
            # State-to-symbol format (default)
            return cls._parse_state_to_symbol(program_text)

    @classmethod
    def _parse_standard_format(cls, program_text):
        """Parse program in standard/compact format: "1RB1RE_1LC0LC_1RD1LB_1RA0RD_---0RC" """
        sections = program_text.split("_")
        actions = []

        for section in sections:
            # Parse actions in groups of 3 characters
            i = 0
            while i < len(section):
                action_str = section[i : i + 3]
                i += 3

                if action_str == "---":
                    # Special halt state
                    actions.append(
                        Action(25, False, -1)
                    )  # Using 25 to represent halt state
                    continue

                if len(action_str) != 3:
                    raise ValueError(f"Invalid action format: {action_str}")

                next_symbol = action_str[0] == "1"

                if action_str[1] == "L":
                    direction = -1
                elif action_str[1] == "R":
                    direction = 1
                else:
                    raise ValueError(f"Invalid direction in action: {action_str}")

                next_state = ord(action_str[2]) - ord("A")
                if next_state < 0 or next_state > 25:
                    raise ValueError(f"Invalid next state in action: {action_str}")

                actions.append(Action(next_state, next_symbol, direction))

        return cls(len(sections), actions)

    @classmethod
    def _parse_state_to_symbol(cls, program_text):
        """Parse program in state-to-symbol format:
        ```
        0	1
        A	1RB	1RE
        B	1LC	0LC
        ...
        ```
        """
        lines = [line.strip() for line in program_text.splitlines() if line.strip()]
        symbol_count = 0
        actions = []
        state_count = 0

        # Skip initial lines until we find a header line
        header_found = False
        for i, line in enumerate(lines):
            if not header_found:
                if line.startswith(("0", "1")):
                    header_found = True
                continue

        # Process state rows
        for line in lines[i + 1 :]:
            parts = line.split()
            if not parts:
                continue

            # First part should be the state letter
            state = parts[0]
            if len(state) != 1 or not "A" <= state <= "Z":
                raise ValueError(f"Invalid state: {state}")

            # Process actions for each symbol
            state_actions = []
            for action_str in parts[1:]:
                if action_str == "---":
                    # Special halt state
                    state_actions.append(Action(25, False, -1))
                    continue

                if len(action_str) != 3:
                    raise ValueError(f"Invalid action format: {action_str}")

                next_symbol = action_str[0] == "1"

                if action_str[1] == "L":
                    direction = -1
                elif action_str[1] == "R":
                    direction = 1
                else:
                    raise ValueError(f"Invalid direction in action: {action_str}")

                next_state = ord(action_str[2]) - ord("A")
                if next_state < 0 or next_state > 25:
                    raise ValueError(f"Invalid next state in action: {action_str}")

                state_actions.append(Action(next_state, next_symbol, direction))

            actions.extend(state_actions)
            symbol_count = max(symbol_count, len(state_actions))
            state_count += 1

        return cls(state_count, actions)

    @classmethod
    def _parse_symbol_to_state(cls, program_text):
        """Parse program in symbol-to-state format (transposed matrix):
        ```
            A	B	C	D	E
        0	1RB	1LC	1RD	1RA	---
        1	1RE	0LC	1LB	0RD	0RC
        ```
        """
        lines = [line.strip() for line in program_text.splitlines() if line.strip()]

        # First, parse the header row to get state names
        header_parts = lines[0].split()
        state_count = len(header_parts)

        # Parse symbol rows
        symbol_rows = []
        for line in lines[1:]:
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            # First part should be the symbol (0 or 1)
            try:
                symbol = int(parts[0])
                if symbol not in (0, 1):
                    raise ValueError(f"Symbol must be 0 or 1, got: {symbol}")
            except ValueError:
                raise ValueError(f"Invalid symbol: {parts[0]}")

            # Process actions for each state
            symbol_actions = []
            for action_str in parts[1:]:
                if action_str == "---":
                    # Special halt state
                    symbol_actions.append(Action(25, False, -1))
                    continue

                if len(action_str) != 3:
                    raise ValueError(f"Invalid action format: {action_str}")

                next_symbol = action_str[0] == "1"

                if action_str[1] == "L":
                    direction = -1
                elif action_str[1] == "R":
                    direction = 1
                else:
                    raise ValueError(f"Invalid direction in action: {action_str}")

                next_state = ord(action_str[2]) - ord("A")
                if next_state < 0 or next_state > 25:
                    raise ValueError(f"Invalid next state in action: {action_str}")

                symbol_actions.append(Action(next_state, next_symbol, direction))

            symbol_rows.append(symbol_actions)

        # Transpose the matrix from symbol-to-state to state-to-symbol format
        actions = []
        for state_idx in range(
            state_count - 1
        ):  # -1 because header row doesn't count as a state
            for symbol_row in symbol_rows:
                if state_idx < len(symbol_row):
                    actions.append(symbol_row[state_idx])

        return cls(
            state_count - 1, actions
        )  # -1 because header row doesn't count as a state


class Machine:
    def __init__(self, program):
        self.state = 0
        self.tape_index = 0
        self.tape = Tape()
        self.program = Program.from_text(program)

    def __iter__(self):
        return self

    def __next__(self):
        # Get the current symbol from the tape
        input_symbol = self.tape[self.tape_index]

        # Get the corresponding action
        action = self.program.action(self.state, input_symbol)

        # Update the tape with the new symbol
        self.tape[self.tape_index] = action.next_symbol

        # Store the previous position to return it
        previous_index = self.tape_index

        # Move the tape head
        self.tape_index += action.direction

        # Update the machine state
        self.state = action.next_state

        # Check if the machine has halted
        if self.state < self.program.state_count:
            return previous_index
        else:
            raise StopIteration

    def is_halted(self):
        return self.state >= self.program.state_count

    def count_ones(self):
        """Count the number of 1s on the tape"""
        return self.tape.count_ones()


def test_machine_halting():
    """Test that a specific machine halts after the expected number of steps with the correct number of ones."""
    import itertools

    machine = Machine("1RB1LC_0LA0LD_1LA1RZ_1LB1RE_0RD0RB")
    step_limit = 10_000_000

    # Count steps until halting or limit
    steps_run = sum(1 for _ in itertools.islice(machine, step_limit))
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
    program_text = """0	1
A	1RB	0RE
B	0RC	0RA
C	1LD	---
D	1LA	0LB
E	1RA	0LC"""

    machine = Machine(program_text)
    step_limit = 100_000

    # Run for up to the step limit
    steps_run = sum(1 for _ in itertools.islice(machine, step_limit))
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
    program_text = """    A   B   C   D   E
0   1RB 1LC 1LA 1RA 1LE
1   1RE 1RD 0LC --- 0RB"""

    machine = Machine(program_text)
    step_limit = 10

    # Run for exactly 10 steps
    steps_run = sum(1 for _ in itertools.islice(machine, step_limit))
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
