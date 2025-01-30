use core::fmt;
use derive_more::derive::{Display, Error, From};
use std::{
    ops::{Index, IndexMut},
    str::FromStr,
};
use thousands::Separable;

const STATE_COUNT: usize = 5;

// Don't change these constants
const STATE_COUNT_U8: u8 = STATE_COUNT as u8;
const SYMBOL_COUNT: usize = 2;

fn main() {
    let program = &Program::from_str(CHAMP_STRING).unwrap();

    let mut machine = Machine {
        tape: Tape::default(),
        tape_index: 0,
        program,
        state: 0,
    };

    let debug_interval = 10_000_000;
    let step_count = machine.debug_count(debug_interval);

    println!(
        "Final: Step {}: {:?}, #1's {}",
        step_count.separate_with_commas(),
        machine,
        machine.tape.count_ones()
    );
}

const CHAMP_STRING: &str = "
    A	B	C	D	E
0	1RB	1RC	1RD	1LA	1RH
1	1LC	1RB	0LE	1LD	0LA
";

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
impl FromStr for Program {
    type Err = CmkError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let mut lines = input.lines();

        // Skip empty lines at the beginning
        for line in lines.by_ref() {
            if !line.trim().is_empty() {
                break;
            }
        }

        let mut vec_of_vec: Vec<Vec<PerInput>> = lines
            .enumerate()
            .map(|(value, line)| {
                let mut parts = line.split_whitespace();

                let value_again = parts.next().ok_or(CmkError::MissingField)?.parse::<u8>()?;
                assert_eq!(value, value_again as usize);

                parts
                    .map(|part| {
                        let next_value =
                            part.chars()
                                .nth(0)
                                .ok_or(CmkError::MissingField)?
                                .to_digit(10)
                                .ok_or(CmkError::InvalidChar)? as u8;

                        let direction = match part.chars().nth(1).ok_or(CmkError::MissingField)? {
                            'L' => Direction::Left,
                            'R' => Direction::Right,
                            _ => return Err(CmkError::InvalidChar),
                        };

                        let next_state = (part
                            .chars()
                            .nth(2)
                            .ok_or(CmkError::MissingField)?
                            .to_digit(36)
                            .ok_or(CmkError::InvalidChar)?
                            .checked_sub('A'.to_digit(36).unwrap())
                            .ok_or(CmkError::InvalidChar)?)
                            as u8;

                        Ok(PerInput {
                            next_state,
                            next_value,
                            direction,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>() // Collect and propagate any errors
            })
            .collect::<Result<Vec<_>, _>>()?; // Collect and propagate errors

        // Ensure proper dimensions (2 x STATE_COUNT)
        if vec_of_vec.len() != SYMBOL_COUNT {
            return Err(CmkError::InvalidLength);
        }
        if vec_of_vec[0].len() != STATE_COUNT {
            return Err(CmkError::InvalidLength);
        }

        // Convert to fixed-size array
        let (row_0, row_1) = vec_of_vec.split_at_mut(1);

        let program: [[PerInput; 2]; STATE_COUNT] =
            row_0[0] // First row
                .drain(..) // Moves out of the first vector
                .zip(row_1[0].drain(..)) // Moves out of the second vector
                .map(|(a, b)| [a, b]) // Create fixed-size arrays
                .collect::<Vec<_>>() // Collect into Vec<[PerInput; 2]>
                .try_into() // Try to convert Vec into [[PerInput; 2]; STATE_COUNT]
                .map_err(|_vec: Vec<[PerInput; 2]>| CmkError::ArrayConversionError)?; // Map error properly

        Ok(Program(program))
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

/// A trait for iterators that can print debug output at intervals.
pub trait DebuggableIterator: Iterator {
    /// Runs the iterator while printing debug output at intervals.
    #[inline]
    fn debug_count(&mut self, debug_interval: usize) -> usize
    where
        Self: Sized + std::fmt::Debug, // ✅ Ensure Debug is implemented
    {
        let mut step_count = 0;

        while let Some(_) = self.next() {
            // ✅ Works for any iterator
            if step_count % debug_interval == 0 {
                println!("Step {}: {:?}", step_count.separate_with_commas(), self);
            }
            step_count += 1;
        }

        step_count
    }
}

// Implement the trait for all Iterators
impl<T> DebuggableIterator for T where T: Iterator + std::fmt::Debug {}

/// Error type for parsing a `Program` from a string.
#[derive(Debug, Display, Error, From)]
pub enum CmkError {
    // #[display(fmt = "Invalid number format: {}", _0)]
    ParseIntError(std::num::ParseIntError),

    // #[display(fmt = "Invalid character encountered in part: '{}'", _0)]
    InvalidChar, // (char),

    // #[display(fmt = "Unexpected empty field in input")]
    MissingField,

    // #[display(fmt = "Unexpected input length")]
    InvalidLength,

    // #[display(fmt = "Failed to convert to array: {:?}", _0)]
    ArrayConversionError,
}
