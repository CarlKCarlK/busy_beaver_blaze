use core::fmt;
use derive_more::Error as DeriveError;
use derive_more::derive::Display;
use std::{
    ops::{Index, IndexMut},
    str::FromStr,
};
use thousands::Separable;
use wasm_bindgen::prelude::*;

const SYMBOL_COUNT: usize = 2;

const BB4_CHAMP: &str = "
	A	B	C	D
0	1RB	1LA	1RH	1RD
1	1LB	0LC	1LD	0RA
";

const BB5_CHAMP: &str = "
    A	B	C	D	E
0	1RB	1RC	1RD	1LA	1RH
1	1LC	1RB	0LE	1LD	0LA
";

const BB6_CONTENDER: &str = "
    	A	B	C	D	E	F
0	1RB	1RC	1LC	0LE	1LF	0RC
1	0LD	0RF	1LA	1RH	0RB	0RE
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

struct Machine<const STATE_COUNT: usize> {
    state: u8,
    tape_index: i32,
    tape: Tape,
    program: Program<STATE_COUNT>,
}

// #[wasm_bindgen]
// impl Machine {
//     #[wasm_bindgen(constructor)]
//     pub fn from_string(input: &str) -> Result<Machine, Error> {
//         input.parse()
//     }

//     #[wasm_bindgen]
//     pub fn step(&mut self) -> Option<()> {
//         self.next()
//     }
// }
impl<const STATE_COUNT: usize> FromStr for Machine<STATE_COUNT> {
    type Err = Error;

    #[allow(clippy::assertions_on_constants)]
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let program: Program<STATE_COUNT> = input.parse()?;

        Ok(Machine {
            tape: Tape::default(),
            tape_index: 0,
            program,
            state: 0,
        })
    }
}

impl<const STATE_COUNT: usize> fmt::Debug for Machine<STATE_COUNT> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Machine {{ state: {}, tape_index: {}}}",
            self.state, self.tape_index
        )
    }
}

impl<const STATE_COUNT: usize> Iterator for Machine<STATE_COUNT> {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        let input = self.tape[self.tape_index];
        let per_input = &self.program.0[self.state as usize][input as usize];
        self.tape[self.tape_index] = per_input.next_symbol;
        self.tape_index += match per_input.direction {
            Direction::Left => -1,
            Direction::Right => 1,
        };
        self.state = per_input.next_state;
        (per_input.next_state < STATE_COUNT as u8).then_some(())
    }
}

#[derive(Debug)]
struct Program<const STATE_COUNT: usize>([[PerInput; SYMBOL_COUNT]; STATE_COUNT]);

// impl a parser for the strings like this
// "   A	B	C	D	E
// 0	1RB	1RC	1RD	1LA	1RH
// 1	1LC	1RB	0LE	1LD	0LA"
impl<const STATE_COUNT: usize> FromStr for Program<STATE_COUNT> {
    type Err = Error;

    #[allow(clippy::assertions_on_constants)]
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let mut lines = input.lines();

        // Skip empty lines at the beginning
        for line in lines.by_ref() {
            if !line.trim().is_empty() {
                break;
            }
        }

        // Create a vector of vectors, e.g. 2 x 5
        let mut vec_of_vec: Vec<Vec<PerInput>> = lines
            .enumerate()
            .map(|(symbol, line)| {
                let mut parts = line.split_whitespace();

                let symbol_again = parts.next().ok_or(Error::MissingField)?.parse::<u8>()?;
                if symbol != symbol_again as usize {
                    return Err(Error::UnexpectedSymbol);
                }

                parts
                    .map(|part| {
                        let asciis = part.as_bytes();
                        if asciis.len() != 3 {
                            return Err(Error::MissingField);
                        }
                        let next_symbol = match asciis[0] {
                            b'0' => 0,
                            b'1' => 1,
                            _ => return Err(Error::InvalidChar),
                        };
                        let direction = match asciis[1] {
                            b'L' => Direction::Left,
                            b'R' => Direction::Right,
                            _ => return Err(Error::InvalidChar),
                        };
                        let next_state = match asciis[2] {
                            b'A'..=b'Z' => asciis[2] - b'A',
                            _ => return Err(Error::InvalidChar),
                        };

                        Ok(PerInput {
                            next_state,
                            next_symbol,
                            direction,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>() // Collect and propagate any errors
            })
            .collect::<Result<Vec<_>, _>>()?; // Collect and propagate errors

        // Ensure proper dimensions (2 x STATE_COUNT)
        if vec_of_vec.len() != SYMBOL_COUNT {
            return Err(Error::InvalidSymbolsCount {
                expected: SYMBOL_COUNT,
                got: vec_of_vec.len(),
            });
        }

        if vec_of_vec[0].len() != STATE_COUNT {
            return Err(Error::InvalidStatesCount {
                expected: STATE_COUNT,
                got: vec_of_vec[0].len(),
            });
        }

        // Convert to fixed-size array
        debug_assert!(SYMBOL_COUNT == 2);
        let (row_0, row_1) = vec_of_vec.split_at_mut(1);

        let program: [[PerInput; 2]; STATE_COUNT] =
            row_0[0] // First row
                .drain(..) // Moves out of the first vector
                .zip(row_1[0].drain(..)) // Moves out of the second vector
                .map(|(a, b)| [a, b]) // Create fixed-size arrays
                .collect::<Vec<_>>() // Collect into Vec<[PerInput; 2]>
                .try_into() // Try to convert Vec into [[PerInput; 2]; STATE_COUNT]
                .map_err(|_vec: Vec<[PerInput; 2]>| Error::ArrayConversionError)?; // Map error properly

        Ok(Program(program))
    }
}

#[derive(Debug)]
struct PerInput {
    next_state: u8,
    next_symbol: u8,
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

        step_count + 1
    }
}

// Implement the trait for all Iterators
impl<T> DebuggableIterator for T where T: Iterator + std::fmt::Debug {}

/// Error type for parsing a `Program` from a string.
#[derive(Debug, Display, DeriveError)]
pub enum Error {
    #[display("Invalid number format: {}", _0)]
    ParseIntError(std::num::ParseIntError),

    #[display("Invalid character encountered in part")]
    InvalidChar,

    #[display("Unexpected empty field in input")]
    MissingField,

    #[display("Unexpected symbols count. Expected {} and got {}", expected, got)]
    InvalidSymbolsCount { expected: usize, got: usize },

    #[display("Unexpected states count. Expected {} and got {}", expected, got)]
    InvalidStatesCount { expected: usize, got: usize },

    #[display("Failed to convert to array")]
    ArrayConversionError,

    #[display("Unexpected symbol encountered")]
    UnexpectedSymbol,
}

// Implement conversions manually where needed
impl From<std::num::ParseIntError> for Error {
    fn from(err: std::num::ParseIntError) -> Self {
        Error::ParseIntError(err)
    }
}

// tests
#[cfg(test)]
mod tests {
    use wasm_bindgen_test::wasm_bindgen_test;
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
    use super::*;

    #[wasm_bindgen_test]
    #[test]
    fn bb5_champ() -> Result<(), Error> {
        let mut machine: Machine<5> = BB5_CHAMP.parse()?;

        let debug_interval = 10_000_000;
        let step_count = machine.debug_count(debug_interval);

        println!(
            "Final: Steps {}: {:?}, #1's {}",
            step_count.separate_with_commas(),
            machine,
            machine.tape.count_ones()
        );

        assert_eq!(step_count, 47_176_870);
        assert_eq!(machine.tape.count_ones(), 4098);
        assert_eq!(machine.state, 7);
        assert_eq!(machine.tape_index, -12242);

        Ok(())
    }
}
