use core::fmt;
use std::ops::{Index, IndexMut};

fn main() {
    let program = &Program::from_string(CHAMP_STRING);

    let mut machine = Machine {
        tape: Tape::default(),
        tape_index: 0,
        program,
        state: 0,
    };

    let mut step_count = 0;
    let mod_base = 10_000;
    while machine.state < 5 {
        if step_count % mod_base == 0 {
            println!("Step: {step_count}: Machine {machine:?}");
        }
        machine.step();
        step_count += 1;
    }
    println!("Final: Step {}: {:?}", step_count, machine);
}

#[derive(Default, Debug)]
struct Tape {
    nonnegative: Vec<u8>,
    negative: Vec<u8>,
}

// Immutable access with `[]`
impl Index<i32> for Tape {
    type Output = u8;

    fn index(&self, index: i32) -> &u8 {
        if index >= 0 {
            self.nonnegative.get(index as usize).unwrap_or(&0)
        } else {
            self.negative.get((-index - 1) as usize).unwrap_or(&0)
        }
    }
}

// Mutable access with `[]`, ensuring growth
impl IndexMut<i32> for Tape {
    fn index_mut(&mut self, index: i32) -> &mut u8 {
        let (index, vec) = if index >= 0 {
            (index as usize, &mut self.nonnegative)
        } else {
            ((-index - 1) as usize, &mut self.negative)
        };

        if vec.len() <= index {
            vec.resize(index + 1, 0);
        }

        &mut vec[index]
    }
}
struct Machine<'a> {
    state: u8,
    tape_index: i32,
    tape: Tape,
    program: &'a Program,
}

impl fmt::Debug for Machine<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Machine {{ state: {}, tape_index: {}}}",
            self.state, self.tape_index
        )
    }
}

// define a step method for the machine
impl Machine<'_> {
    fn step(&mut self) {
        let input = self.tape[self.tape_index];
        // println!(
        //     "Debug: state={}, head={}, reading={}",
        //     self.state, self.tape_index, input
        // );
        let per_input = &self.program.0[self.state as usize].0[input as usize];
        self.state = per_input.next_state;
        self.tape[self.tape_index] = per_input.next_value;
        match per_input.direction {
            Direction::Left => self.tape_index -= 1,
            Direction::Right => self.tape_index += 1,
        }
    }
}

#[derive(Debug)]
struct Program([PerState; 5]);

// impl a parser for the strings like this
// "   A	B	C	D	E
// 0	1RB	1RC	1RD	1LA	1RH
// 1	1LC	1RB	0LE	1LD	0LA"
impl Program {
    fn from_string(input: &str) -> Self {
        let mut lines = input.lines();
        let _header = lines.next().unwrap();
        let mut vec_of_vec = lines
            .enumerate()
            .map(|(value, line)| {
                let mut parts = line.split_whitespace();
                // println!("Line {}: {:?}", value, line);
                let value_again = parts.next().unwrap().parse::<u8>().unwrap();
                assert_eq!(value, value_again as usize);
                parts
                    .enumerate()
                    .map(|(_state, part)| {
                        // println!("Part: {:?}", part);
                        let next_value = part.chars().nth(0).unwrap() as u8 - b'0';
                        let direction = match part.chars().nth(1).unwrap() {
                            'L' => Direction::Left,
                            'R' => Direction::Right,
                            _ => panic!("Invalid direction"),
                        };
                        let next_state = part.chars().nth(2).unwrap() as u8 - b'A';
                        PerInput {
                            next_state,
                            next_value,
                            direction,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // Turn 2 x 5 vec of vecs into 5 x 2 vec of arrays

        Program(
            (0..vec_of_vec[0].len()) // Iterate over 5 states
                .map(|_state| {
                    PerState([
                        vec_of_vec[0].remove(0), // Remove first item, shifting the rest left
                        vec_of_vec[1].remove(0), // Remove first item, shifting the rest left
                    ])
                })
                .collect::<Vec<_>>() // Collect into Vec<PerState>
                .try_into() // Convert Vec<PerState> into [PerState; 5]
                .unwrap(), // Ensure length is exactly 5
        )
    }
}

#[derive(Debug)]
struct PerState([PerInput; 2]);

#[derive(Debug)]
struct PerInput {
    next_state: u8,
    next_value: u8,
    direction: Direction,
}

#[derive(Debug)]
enum Direction {
    Left,
    Right,
}

static PROGRAM_1: Program = Program([
    PerState([
        PerInput {
            next_state: 1, // A -> B
            next_value: 1,
            direction: Direction::Right,
        },
        PerInput {
            next_state: 2, // A -> C
            next_value: 1,
            direction: Direction::Left,
        },
    ]),
    PerState([
        PerInput {
            next_state: 2, // B -> C
            next_value: 1,
            direction: Direction::Right,
        },
        PerInput {
            next_state: 1, // B -> B
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
    PerState([
        PerInput {
            next_state: 3, // C -> D
            next_value: 1,
            direction: Direction::Left,
        },
        PerInput {
            next_state: 4, // C -> E
            next_value: 0,
            direction: Direction::Left,
        },
    ]),
    PerState([
        PerInput {
            next_state: 4, // D -> E
            next_value: 1,
            direction: Direction::Right,
        },
        PerInput {
            next_state: 0, // D -> A
            next_value: 1,
            direction: Direction::Left,
        },
    ]),
    PerState([
        PerInput {
            next_state: 5, // E -> Halt
            next_value: 1,
            direction: Direction::Left,
        },
        PerInput {
            next_state: 3, // E -> D
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
]);

static PROGRAM_2: Program = Program([
    // State 0 (think of it like 'A')
    PerState([
        // If tape cell = 0
        PerInput {
            next_state: 1, // Go to state 1 (B)
            next_value: 1, // Write '1'
            direction: Direction::Right,
        },
        // If tape cell = 1
        PerInput {
            next_state: 2, // Go to state 2 (C)
            next_value: 1, // Write '1'
            direction: Direction::Right,
        },
    ]),
    // State 1 (B)
    PerState([
        // If tape cell = 0
        PerInput {
            next_state: 3, // Go to state 3 (D)
            next_value: 1, // Write '1'
            direction: Direction::Left,
        },
        // If tape cell = 1
        PerInput {
            next_state: 1, // Stay in state B
            next_value: 1, // Write '1' again
            direction: Direction::Right,
        },
    ]),
    // State 2 (C)
    PerState([
        // If tape cell = 0
        PerInput {
            next_state: 4, // Go to state 4 (E)
            next_value: 1, // Write '1'
            direction: Direction::Right,
        },
        // If tape cell = 1
        PerInput {
            next_state: 1, // Jump back to state B
            next_value: 0, // Write '0'
            direction: Direction::Left,
        },
    ]),
    // State 3 (D)
    PerState([
        // If tape cell = 0
        PerInput {
            next_state: 2, // Go to state C
            next_value: 1, // Write '1'
            direction: Direction::Right,
        },
        // If tape cell = 1
        PerInput {
            next_state: 0, // Jump back to state A (0)
            next_value: 1, // Write '1'
            direction: Direction::Left,
        },
    ]),
    // State 4 (E)
    PerState([
        // If tape cell = 0
        PerInput {
            next_state: 5, // 5 means HALT in your setup
            next_value: 1, // Write '1'
            direction: Direction::Left,
        },
        // If tape cell = 1
        PerInput {
            next_state: 3, // Go to state D
            next_value: 1, // Write '1'
            direction: Direction::Right,
        },
    ]),
]);

static PROGRAM_QUICKER: Program = Program([
    // State 0 (A)
    PerState([
        // (read 0) -> write 1, move Right, go to state=1
        PerInput {
            next_state: 1,
            next_value: 1,
            direction: Direction::Right,
        },
        // (read 1) -> write 1, move Left, go to state=2
        PerInput {
            next_state: 2,
            next_value: 1,
            direction: Direction::Left,
        },
    ]),
    // State 1 (B)
    PerState([
        // (read 0) -> write 1, move Left, go to state=3
        PerInput {
            next_state: 3,
            next_value: 1,
            direction: Direction::Left,
        },
        // (read 1) -> write 1, move Right, go to state=1 (loop)
        PerInput {
            next_state: 1,
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
    // State 2 (C)
    PerState([
        // (read 0) -> write 1, move Right, go to state=4
        PerInput {
            next_state: 4,
            next_value: 1,
            direction: Direction::Right,
        },
        // (read 1) -> write 0, move Left,  go to state=1
        PerInput {
            next_state: 1,
            next_value: 0,
            direction: Direction::Left,
        },
    ]),
    // State 3 (D)
    PerState([
        // (read 0) -> write 1, move Right, go to state=2
        PerInput {
            next_state: 2,
            next_value: 1,
            direction: Direction::Right,
        },
        // (read 1) -> write 1, move Left,  go to state=0
        PerInput {
            next_state: 0,
            next_value: 1,
            direction: Direction::Left,
        },
    ]),
    // State 4 (E)
    PerState([
        // (read 0) -> write 1, move Left,  go to state=5 (halt)
        PerInput {
            next_state: 5,
            next_value: 1,
            direction: Direction::Left,
        },
        // (read 1) -> write 1, move Right, go to state=3
        PerInput {
            next_state: 3,
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
]);

static PROGRAM_QUICK_CHECK: Program = Program([
    // ---------- State 0 (A) ----------
    PerState([
        // If tape cell = 0
        PerInput {
            next_state: 1, // go to B
            next_value: 1, // write 1
            direction: Direction::Right,
        },
        // If tape cell = 1
        PerInput {
            next_state: 3, // go to D
            next_value: 0, // write 0
            direction: Direction::Left,
        },
    ]),
    // ---------- State 1 (B) ----------
    PerState([
        // If tape cell = 0
        PerInput {
            next_state: 2, // go to C
            next_value: 1, // write 1
            direction: Direction::Right,
        },
        // If tape cell = 1
        PerInput {
            next_state: 1, // stay in B
            next_value: 1, // write 1 again
            direction: Direction::Right,
        },
    ]),
    // ---------- State 2 (C) ----------
    PerState([
        // If tape cell = 0
        PerInput {
            next_state: 1, // go back to B
            next_value: 1, // write 1
            direction: Direction::Left,
        },
        // If tape cell = 1
        PerInput {
            next_state: 4, // go to E
            next_value: 1, // write 1
            direction: Direction::Right,
        },
    ]),
    // ---------- State 3 (D) ----------
    PerState([
        // If tape cell = 0
        PerInput {
            next_state: 4, // go to E
            next_value: 1, // write 1
            direction: Direction::Left,
        },
        // If tape cell = 1
        PerInput {
            next_state: 0, // go back to A
            next_value: 1, // write 1
            direction: Direction::Right,
        },
    ]),
    // ---------- State 4 (E) ----------
    PerState([
        // If tape cell = 0
        PerInput {
            next_state: 5, // 5 = HALT
            next_value: 1, // write 1
            direction: Direction::Left,
        },
        // If tape cell = 1
        PerInput {
            next_state: 3, // return to D
            next_value: 1, // write 1
            direction: Direction::Right,
        },
    ]),
]);

static PROGRAM_SUPER_SIMPLE: Program = Program([
    // State 0 (A)
    PerState([
        // Reading 0 -> next_state=1, write=1, move=R
        PerInput {
            next_state: 1,
            next_value: 1,
            direction: Direction::Right,
        },
        // Reading 1 -> next_state=1, write=1, move=R
        PerInput {
            next_state: 1,
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
    // State 1 (B)
    PerState([
        // Reading 0
        PerInput {
            next_state: 2,
            next_value: 1,
            direction: Direction::Right,
        },
        // Reading 1
        PerInput {
            next_state: 2,
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
    // State 2 (C)
    PerState([
        // Reading 0
        PerInput {
            next_state: 3,
            next_value: 1,
            direction: Direction::Right,
        },
        // Reading 1
        PerInput {
            next_state: 3,
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
    // State 3 (D)
    PerState([
        // Reading 0
        PerInput {
            next_state: 4,
            next_value: 1,
            direction: Direction::Right,
        },
        // Reading 1
        PerInput {
            next_state: 4,
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
    // State 4 (E)
    PerState([
        // Reading 0 -> next_state=5 (halt), write=1, move=R
        PerInput {
            next_state: 5,
            next_value: 1,
            direction: Direction::Right,
        },
        // Reading 1 -> next_state=5 (halt), write=1, move=R
        PerInput {
            next_state: 5,
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
]);

static PROGRAM_LITTLE_MORE: Program = Program([
    // State 0 (A)
    PerState([
        // Reading 0 -> next_state=1, write=1, move=Right
        PerInput {
            next_state: 1,
            next_value: 1,
            direction: Direction::Right,
        },
        // Reading 1 -> next_state=1, write=1, move=Right
        PerInput {
            next_state: 1,
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
    // State 1 (B)
    PerState([
        // Reading 0 -> next_state=2, write=1, move=Left
        PerInput {
            next_state: 2,
            next_value: 1,
            direction: Direction::Left,
        },
        // Reading 1 -> next_state=2, write=1, move=Left
        PerInput {
            next_state: 2,
            next_value: 1,
            direction: Direction::Left,
        },
    ]),
    // State 2 (C)
    PerState([
        // Reading 0 -> next_state=3, write=1, move=Right
        PerInput {
            next_state: 3,
            next_value: 1,
            direction: Direction::Right,
        },
        // Reading 1 -> next_state=3, write=1, move=Right
        PerInput {
            next_state: 3,
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
    // State 3 (D)
    PerState([
        // Reading 0 -> next_state=4, write=1, move=Left
        PerInput {
            next_state: 4,
            next_value: 1,
            direction: Direction::Left,
        },
        // Reading 1 -> next_state=4, write=1, move=Left
        PerInput {
            next_state: 4,
            next_value: 1,
            direction: Direction::Left,
        },
    ]),
    // State 4 (E)
    PerState([
        // Reading 0 -> next_state=5 (HALT), write=1, move=Right
        PerInput {
            next_state: 5,
            next_value: 1,
            direction: Direction::Right,
        },
        // Reading 1 -> next_state=5 (HALT), write=1, move=Right
        PerInput {
            next_state: 5,
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
]);

static PROGRAM_BB5_CHAMPION: Program = Program([
    // State A (0)
    PerState([
        // If reading 0: write 1, move Right, go to state B (1)
        PerInput {
            next_state: 1,
            next_value: 1,
            direction: Direction::Right,
        },
        // If reading 1: write 1, move Left, go to state C (2)
        PerInput {
            next_state: 2,
            next_value: 1,
            direction: Direction::Left,
        },
    ]),
    // State B (1)
    PerState([
        // If reading 0: write 1, move Right, go to state C (2)
        PerInput {
            next_state: 2,
            next_value: 1,
            direction: Direction::Right,
        },
        // If reading 1: write 1, move Right, stay in state B (1)
        PerInput {
            next_state: 1,
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
    // State C (2)
    PerState([
        // If reading 0: write 1, move Left, go to state D (3)
        PerInput {
            next_state: 3,
            next_value: 1,
            direction: Direction::Left,
        },
        // If reading 1: write 0, move Left, go to state E (4)
        PerInput {
            next_state: 4,
            next_value: 0,
            direction: Direction::Left,
        },
    ]),
    // State D (3)
    PerState([
        // If reading 0: write 1, move Right, go to state E (4)
        PerInput {
            next_state: 4,
            next_value: 1,
            direction: Direction::Right,
        },
        // If reading 1: write 1, move Left,  go back to state A (0)
        PerInput {
            next_state: 0,
            next_value: 1,
            direction: Direction::Left,
        },
    ]),
    // State E (4)
    PerState([
        // If reading 0: write 1, move Left,  go to Halt (5)
        PerInput {
            next_state: 5,
            next_value: 1,
            direction: Direction::Left,
        },
        // If reading 1: write 1, move Right, go to state D (3)
        PerInput {
            next_state: 3,
            next_value: 1,
            direction: Direction::Right,
        },
    ]),
]);

const CHAMP_STRING: &str = "   A	B	C	D	E
0	1RB	1RC	1RD	1LA	1RH
1	1LC	1RB	0LE	1LD	0LA
";
