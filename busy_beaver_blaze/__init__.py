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
        """Parse a program from text representation in standard format.
        Format (standard): Compact one-line format with states separated by underscores
        Example: "1RB1RE_1LC0LC_1RD1LB_1RA0RD_---0RC"
        """
        program_text = program_text.strip()

        # Determine which format is being used
        if "_" in program_text and program_text.count("\n") == 0:
            # Standard format (one line with underscores)
            return cls._parse_standard_format(program_text)
        else:
            raise ValueError("Invalid program format. Expected standard format.")

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
