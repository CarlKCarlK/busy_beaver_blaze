use core::fmt;
use std::ops::{Index, IndexMut};

use thousands::Separable;

const STATE_COUNT: usize = 5;

// Don't change these constants
const STATE_COUNT_U8: u8 = STATE_COUNT as u8;
const SYMBOL_COUNT: usize = 2;

fn main() {
    let program = &Program::from_string(CHAMP_STRING);

    let mut machine = Machine {
        tape: Tape::default(),
        tape_index: 0,
        program,
        state: 0,
    };

    let mod_base = 10_000;
    for (step_count, _) in (&mut machine).enumerate() {
        if step_count % mod_base == 0 {
            println!(
                "Step: {}: Machine {machine:?}",
                step_count.separate_with_commas()
            );
        }
    }
    println!(
        "Final: Step {}: {:?}, #1's {}",
        step_count.separate_with_commas(),
        machine,
        machine.tape.count_ones()
    );
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
impl Tape {
    fn count_ones(&self) -> usize {
        self.nonnegative
            .iter()
            .chain(self.negative.iter()) // Combine both vectors
            .map(|&x| (x == 1) as usize)
            .sum()
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

impl Iterator for Machine<'_> {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        let input = self.tape[self.tape_index];
        let per_input = &self.program.0[self.state as usize][input as usize];
        self.tape[self.tape_index] = per_input.next_value;
        self.tape_index += match per_input.direction {
            Direction::Left => -1,
            Direction::Right => 1,
        };
        self.state = per_input.next_state;
        (per_input.next_state < STATE_COUNT_U8).then_some(())
    }
}

#[derive(Debug)]
struct Program([[PerInput; SYMBOL_COUNT]; STATE_COUNT]);

// impl a parser for the strings like this
// "   A	B	C	D	E
// 0	1RB	1RC	1RD	1LA	1RH
// 1	1LC	1RB	0LE	1LD	0LA"
impl Program {
    #[allow(clippy::assertions_on_constants)]
    fn from_string(input: &str) -> Self {
        let mut lines = input.lines();
        while lines.next().unwrap() == "" {
            // Skip empty lines
        }
        let mut vec_of_vec = lines
            .enumerate()
            .map(|(value, line)| {
                let mut parts = line.split_whitespace();
                // println!("Line {}: {:?}", value, line);
                let value_again = parts.next().unwrap().parse::<u8>().unwrap();
                assert_eq!(value, value_again as usize);
                parts
                    .map(|part| {
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
        // Turn 2 x STATE_COUNT vec of vec into STATE_COUNT x 2 array of arrays

        assert_eq!(vec_of_vec.len(), SYMBOL_COUNT, "Expected 2 rows");
        debug_assert!(SYMBOL_COUNT == 2, "Expected 2 symbols");
        assert_eq!(
            vec_of_vec[0].len(),
            STATE_COUNT,
            "Expected STATE_COUNT columns"
        );

        Program(
            (0..STATE_COUNT)
                .map(|_state| {
                    [
                        vec_of_vec[0].remove(0), // Remove first item, shifting the rest left
                        vec_of_vec[1].remove(0), // Remove first item, shifting the rest left
                    ]
                })
                .collect::<Vec<_>>() // Collect into Vec<PerState>
                .try_into() // Convert Vec<PerState> into [PerState; STATE_COUNT]
                .unwrap(), // Ensure length is exactly STATE_COUNT
        )
    }
}

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

const CHAMP_STRING: &str = "
    A	B	C	D	E
0	1RB	1RC	1RD	1LA	1RH
1	1LC	1RB	0LE	1LD	0LA
";
