use crate::{Error, Symbol, Tape};
use arrayvec::ArrayVec;
use core::{fmt, str::FromStr};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Machine {
    state: u8,
    tape_index: i64,
    pub(crate) tape: Tape, // TODO make private
    pub(crate) program: Program,
}

#[wasm_bindgen]
impl Machine {
    /// Creates a new Machine from a program string.
    ///
    /// # Errors
    ///
    /// Returns an error if the program string cannot be parsed into a valid Turing machine program.
    #[wasm_bindgen(constructor)]
    pub fn from_string(program: &str) -> Result<Self, String> {
        program.parse().map_err(|error| format!("{error:?}"))
    }

    #[wasm_bindgen]
    pub fn step(&mut self) -> bool {
        self.next().is_some()
    }

    #[wasm_bindgen]
    #[inline]
    #[must_use]
    /// # Panics
    /// Panics if the number of nonblank cells exceeds `u32::MAX`.
    pub fn count_nonblanks(&self) -> u32 {
        u32::try_from(self.tape.count_nonblanks()).expect("count_nonblanks exceeds u32 range")
    }

    #[wasm_bindgen]
    #[inline]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn is_halted(&self) -> bool {
        self.program.state_count <= self.state
    }

    #[wasm_bindgen(js_name = "count")]
    pub fn count_js(&mut self, early_stop_is_some: bool, early_stop_number: u64) -> u64 {
        let mut step_index = 0;

        while self.next().is_some() && (!early_stop_is_some || step_index < early_stop_number) {
            step_index += 1;
        }

        step_index + 1 // turn last index into count
    }

    #[wasm_bindgen]
    #[inline]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn state(&self) -> u8 {
        self.state
    }

    #[wasm_bindgen]
    #[inline]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn tape_index(&self) -> i64 {
        self.tape_index
    }
}

impl Machine {
    #[inline]
    #[must_use]
    pub const fn tape(&self) -> &Tape {
        &self.tape
    }

    #[inline]
    #[must_use]
    pub const fn symbol_count(&self) -> u8 {
        self.program.symbol_count
    }
}

impl FromStr for Machine {
    type Err = Error;

    #[allow(clippy::assertions_on_constants)]
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let program: Program = input.parse()?;

        Ok(Self {
            tape: Tape::default(),
            tape_index: 0,
            program,
            state: 0,
        })
    }
}

impl fmt::Debug for Machine {
    #[allow(clippy::min_ident_chars)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Machine {{ state: {}, tape_index: {}}}",
            self.state, self.tape_index
        )
    }
}

#[allow(clippy::missing_trait_methods)]
impl Iterator for Machine {
    type Item = i64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let program = &self.program;
        let input = self.tape.read(self.tape_index);
        let action = program.action(self.state, input);
        // Always write the next symbol (revert skipped-write optimization)
        self.tape.write(self.tape_index, action.next_symbol);
        let previous_index = self.tape_index;
        self.tape_index += i64::from(action.direction);
        self.state = action.next_state;
        if self.state < program.state_count {
            Some(previous_index)
        } else {
            None
        }
    }
}

type StateToSymbolToAction = ArrayVec<Action, { Program::MAX_STATE_COUNT }>;

#[derive(Debug)]
pub struct Program {
    state_count: u8,
    pub(crate) symbol_count: u8,
    state_to_symbol_to_action: StateToSymbolToAction, // Changed from SmallVec
}

// impl a parser for the strings like this
// "   A	B	C	D	E
// 0	1RB	1RC	1RD	1LA	1RH
// 1	1LC	1RB	0LE	1LD	0LA"
impl FromStr for Program {
    type Err = Error;

    #[allow(clippy::assertions_on_constants, clippy::min_ident_chars)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let count_lines = s.lines().count();
        let is_first_non_space_a_numeral = s.trim().starts_with(|c: char| c.is_ascii_digit());

        match (count_lines, is_first_non_space_a_numeral) {
            (1, _) => Self::parse_standard_format(s),
            (2.., false) => Self::parse_symbol_to_state(s),
            (2.., true) => Self::parse_state_to_symbol(s),
            _ => Err(Error::UnknownProgramFormat),
        }
    }
}

impl Program {
    // pub const SYMBOL_COUNT: usize = 2;
    pub const MAX_STATE_COUNT: usize = 100;

    #[inline]
    fn action(&self, state: u8, symbol: Symbol) -> &Action {
        let offset = (state * self.symbol_count) as usize + usize::from(symbol);
        &self.state_to_symbol_to_action[offset]
    }

    fn parse_state(input: impl AsRef<str>) -> Result<char, Error> {
        let mut chars = input.as_ref().chars();
        match (chars.next(), chars.next()) {
            (Some(char @ 'A'..='Z'), Option::None) => Ok(char), // Ensure single uppercase letter
            _ => Err(Error::UnexpectedState),
        }
    }

    fn parse_action(part: impl AsRef<str>) -> Result<Action, Error> {
        let part = part.as_ref();
        let asciis = part.as_bytes();
        if asciis.len() != 3 {
            return Err(Error::MissingField);
        }
        if asciis == b"---" {
            return Ok(Action {
                next_state: 25,
                next_symbol: Symbol::STATE_ZERO,
                direction: -1,
            });
        }
        let next_symbol = match asciis[0] {
            b'0' => Symbol::STATE_ZERO,
            b'1' => Symbol::STATE_ONE,
            b'2' => Symbol::STATE_TWO,
            b'3' => Symbol::STATE_THREE,
            b'4' => Symbol::STATE_FOUR,
            b'5' => Symbol::STATE_FIVE,
            b'6' => Symbol::STATE_SIX,
            b'7' => Symbol::STATE_SEVEN,
            b'8' => Symbol::STATE_EIGHT,
            b'9' => Symbol::STATE_NINE,
            // cmk catch at compile time?
            _ => {
                return Err(Error::InvalidChar {
                    got: asciis[0] as char,
                });
            }
        };
        let direction = match asciis[1] {
            b'L' => -1,
            b'R' => 1,
            _ => {
                return Err(Error::InvalidChar {
                    got: asciis[1] as char,
                });
            }
        };
        let next_state = match asciis[2] {
            b'A'..=b'Z' => asciis[2] - b'A',
            _ => {
                return Err(Error::InvalidChar {
                    got: asciis[2] as char,
                });
            }
        };

        Ok(Action {
            next_state,
            next_symbol,
            direction,
        })
    }

    #[allow(clippy::assertions_on_constants, clippy::min_ident_chars)]
    fn parse_state_to_symbol(s: &str) -> Result<Self, Error> {
        let mut lines = s.lines();

        // Skip empty lines at the beginning
        for line in lines.by_ref() {
            if !line.trim().is_empty() {
                break;
            }
        }

        // Create a vector of vectors, e.g. 5 x 2
        let state_to_symbol_to_action: Vec<Vec<Action>> = lines
            .zip('A'..)
            .map(|(line, state)| {
                let mut parts = line.split_whitespace();

                let state_again = parts
                    .next()
                    .ok_or(Error::MissingField)
                    .and_then(Self::parse_state)?;

                if state != state_again {
                    return Err(Error::UnexpectedState);
                }

                parts.map(Self::parse_action).collect::<Result<Vec<_>, _>>() // Collect and propagate any errors
            })
            .collect::<Result<Vec<_>, _>>()?; // Collect and propagate errors

        let symbol_count =
            u8::try_from(state_to_symbol_to_action[0].len()).expect("symbol_count must fit in u8");
        if symbol_count == 0 {
            return Err(Error::UnexpectedSymbolCount { got: 0 });
        }

        // Ensure proper dimensions (STATE_COUNT x 2)
        let state_count =
            u8::try_from(state_to_symbol_to_action.len()).expect("state_count must fit in u8");
        if state_count == 0 {
            return Err(Error::InvalidStatesCount {
                expected: 1,
                got: 0,
            });
        }

        if state_count as usize > Self::MAX_STATE_COUNT {
            return Err(Error::InvalidStatesCount {
                expected: Self::MAX_STATE_COUNT,
                got: state_count as usize,
            });
        }

        Ok(Self {
            state_count,
            symbol_count,
            state_to_symbol_to_action: state_to_symbol_to_action.into_iter().flatten().collect(),
        })
    }

    #[allow(
        clippy::assertions_on_constants,
        clippy::min_ident_chars,
        clippy::needless_collect
    )]
    fn parse_standard_format(s: &str) -> Result<Self, Error> {
        let sections = s.trim().split('_');

        // Create a vector of vectors, e.g. 5 x 2
        let state_to_symbol_to_action: Vec<Vec<Action>> = sections
            .zip('A'..)
            .map(|(section, _state)| {
                // split section into groups of 3 char
                let parts: Vec<String> = section
                    .chars()
                    .collect::<Vec<_>>() // Collect into Vec<char>
                    .chunks(3) // Chunk it into groups
                    .map(|chunk| chunk.iter().collect()) // Convert each chunk back into a String
                    .collect();

                parts
                    .into_iter()
                    .map(Self::parse_action)
                    .collect::<Result<Vec<_>, _>>() // Collect and propagate any errors
            })
            .collect::<Result<Vec<_>, _>>()?; // Collect and propagate errors

        // Ensure proper dimensions (STATE_COUNT x 2)
        let state_count =
            u8::try_from(state_to_symbol_to_action.len()).expect("state_count must fit in u8");
        if state_count == 0 {
            return Err(Error::InvalidStatesCount {
                expected: 1,
                got: 0,
            });
        }

        let symbol_count =
            u8::try_from(state_to_symbol_to_action[0].len()).expect("symbol_count must fit in u8");
        if symbol_count == 0 {
            return Err(Error::UnexpectedSymbolCount { got: 0 });
        }

        if state_count as usize > Self::MAX_STATE_COUNT {
            return Err(Error::InvalidStatesCount {
                expected: Self::MAX_STATE_COUNT,
                got: state_count as usize,
            });
        }

        Ok(Self {
            state_count,
            symbol_count,
            state_to_symbol_to_action: state_to_symbol_to_action.into_iter().flatten().collect(),
        })
    }

    #[allow(clippy::assertions_on_constants, clippy::min_ident_chars)]
    fn parse_symbol_to_state(s: &str) -> Result<Self, Error> {
        let mut lines = s.lines();

        // Skip empty lines at the beginning
        for line in lines.by_ref() {
            if !line.trim().is_empty() {
                break;
            }
        }

        // Create a vector of vectors, e.g. 2 x 5
        let vec_of_vec: Vec<Vec<Action>> = lines
            .enumerate()
            .map(|(symbol, line)| {
                let mut parts = line.split_whitespace();

                let symbol_again = parts.next().ok_or(Error::MissingField)?.parse::<u8>()?;
                if symbol != symbol_again as usize {
                    return Err(Error::UnexpectedSymbol);
                }

                parts.map(Self::parse_action).collect::<Result<Vec<_>, _>>() // Collect and propagate any errors
            })
            .collect::<Result<Vec<_>, _>>()?; // Collect and propagate errors

        // Ensure proper dimensions (2 x STATE_COUNT)
        let symbol_count = vec_of_vec.len();
        if symbol_count == 0 {
            return Err(Error::UnexpectedSymbolCount { got: 0 });
        }

        let state_count = vec_of_vec[0].len();
        if state_count == 0 {
            return Err(Error::InvalidStatesCount {
                expected: 1,
                got: 0,
            });
        }

        if state_count > Self::MAX_STATE_COUNT {
            return Err(Error::InvalidStatesCount {
                expected: Self::MAX_STATE_COUNT,
                got: state_count,
            });
        }

        // Preallocate transposed vec_of_vec (state_count x symbol_count)
        let mut state_to_symbol_to_action: Vec<Vec<Action>> = (0..state_count)
            .map(|_| Vec::with_capacity(symbol_count))
            .collect();

        // Drain and fill the transposed matrix
        for row in vec_of_vec {
            if row.len() != state_count {
                return Err(Error::InvalidStatesCount {
                    expected: state_count,
                    got: row.len(),
                });
            }

            for (i, item) in row.into_iter().enumerate() {
                state_to_symbol_to_action[i].push(item); // Move item into transposed[i]
            }
        }

        Ok(Self {
            state_count: u8::try_from(state_count).expect("state_count must fit in u8"),
            symbol_count: u8::try_from(symbol_count).expect("symbol_count must fit in u8"),
            state_to_symbol_to_action: state_to_symbol_to_action.into_iter().flatten().collect(),
        })
    }
}

#[derive(Debug)]
struct Action {
    next_state: u8,
    next_symbol: Symbol,
    direction: i8,
}
