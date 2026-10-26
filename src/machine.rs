use crate::{Error, Symbol, Tape};
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
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
    #[must_use]
    /// # Panics
    /// Panics if the number of nonblank cells exceeds `u32::MAX`.
    pub fn count_nonblanks(&self) -> u32 {
        u32::try_from(self.tape.count_nonblanks()).expect("count_nonblanks exceeds u32 range")
    }

    #[wasm_bindgen]
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
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
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn state(&self) -> u8 {
        self.state
    }

    #[wasm_bindgen]
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
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

type StateToSymbolToAction = Box<[Action]>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    state_count: u8,
    pub(crate) symbol_count: u8,
    state_to_symbol_to_action: StateToSymbolToAction, // Changed from SmallVec, then ArrayVec
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
    /// Maximum number of states named by the uppercase letters A–Z.
    pub const MAX_STATE_COUNT: usize = 26;
    /// Maximum number of symbols encoded by 0–9 and a–z.
    pub const MAX_SYMBOL_COUNT: usize = 36;

    #[inline]
    fn action(&self, state: u8, symbol: Symbol) -> &Action {
        let offset = usize::from(state) * usize::from(self.symbol_count) + usize::from(symbol);
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
                next_state: u8::MAX,
                next_symbol: Symbol::STATE_ZERO,
                direction: -1,
            });
        }
        let next_symbol = match asciis[0] {
            b'0'..=b'9' => Symbol::from(asciis[0] - b'0'),
            b'a'..=b'z' => Symbol::from(asciis[0] - b'a' + 10),
            // TODO catch at compile time?
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

        Self::from_rows(state_to_symbol_to_action)
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

        Self::from_rows(state_to_symbol_to_action)
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

                let label = parts.next().ok_or(Error::MissingField)?;
                let symbol_again = match label.as_bytes() {
                    [symbol_char @ b'a'..=b'z'] => symbol_char - b'a' + 10,
                    _ => label.parse::<u8>()?,
                };
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

        Self::from_rows(state_to_symbol_to_action)
    }

    fn from_rows(rows: Vec<Vec<Action>>) -> Result<Self, Error> {
        let state_count = rows.len();
        if !(1..=Self::MAX_STATE_COUNT).contains(&state_count) {
            return Err(Error::InvalidStatesCount {
                expected: Self::MAX_STATE_COUNT,
                got: state_count,
            });
        }
        let symbol_count = rows[0].len();
        if !(1..=Self::MAX_SYMBOL_COUNT).contains(&symbol_count) {
            return Err(Error::UnexpectedSymbolCount { got: symbol_count });
        }
        for row in &rows {
            if row.len() != symbol_count {
                return Err(Error::UnexpectedSymbolCount { got: row.len() });
            }
            if row
                .iter()
                .any(|action| usize::from(action.next_symbol) >= symbol_count)
            {
                return Err(Error::UnexpectedSymbol);
            }
        }
        // Preserve the historical reported halt state Z for existing programs.
        // A 26-state program uses Z as a working state and needs a separate sentinel.
        let halt_state = if state_count < Self::MAX_STATE_COUNT {
            b'Z' - b'A'
        } else {
            u8::MAX
        };
        Ok(Self {
            state_count: u8::try_from(state_count).expect("state_count must fit in u8"),
            symbol_count: u8::try_from(symbol_count).expect("symbol_count must fit in u8"),
            state_to_symbol_to_action: rows
                .into_iter()
                .flatten()
                .map(|mut action| {
                    if action.next_state == u8::MAX {
                        action.next_state = halt_state;
                    }
                    action
                })
                .collect(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Action {
    next_state: u8,
    next_symbol: Symbol,
    direction: i8,
}

#[cfg(test)]
mod tests {
    use super::*;

    const BB10: &str = "1RB0RA_1LC1LF_1RD0LB_1RA1LE_1LI0LC_0RG1LD_1RH1LG_1LC0RG_0LJ0LI_1RZ0LC";

    #[test]
    fn extended_symbols_in_all_formats() {
        let actions = format!("aRA{}", "0RA".repeat(10));
        let standard = Machine::from_string(&actions).unwrap();
        let symbol_major = format!(
            "A\n{}",
            actions
                .as_bytes()
                .chunks_exact(3)
                .enumerate()
                .map(|(symbol_index, action)| {
                    let label = if symbol_index == 10 {
                        "a".to_owned()
                    } else {
                        symbol_index.to_string()
                    };
                    format!("{label} {}", core::str::from_utf8(action).unwrap())
                })
                .collect::<Vec<_>>()
                .join("\n")
        );
        let state_major = format!(
            "{}\nA {}",
            (0..11)
                .map(|symbol_index| symbol_index.to_string())
                .collect::<Vec<_>>()
                .join(" "),
            actions
                .as_bytes()
                .chunks_exact(3)
                .map(|action| core::str::from_utf8(action).unwrap())
                .collect::<Vec<_>>()
                .join(" ")
        );
        for mut machine in [
            standard,
            Machine::from_string(&symbol_major).unwrap(),
            Machine::from_string(&state_major).unwrap(),
        ] {
            assert!(machine.step());
            assert_eq!(u8::from(machine.tape.read(0)), 10);
            assert_eq!(machine.symbol_count(), 11);
        }
    }

    #[test]
    fn transition_offsets_exceed_u8_and_the_old_array_capacity() {
        let program = (0..11)
            .map(|state_index| {
                let next_state = if state_index == 10 {
                    'Z'
                } else {
                    char::from(b'A' + state_index + 1)
                };
                format!("zR{next_state}").repeat(36)
            })
            .collect::<Vec<_>>()
            .join("_");
        let mut machine = Machine::from_string(&program).unwrap();
        assert_eq!(machine.program.state_to_symbol_to_action.len(), 396);
        for _step_index in 0..10 {
            assert!(machine.step());
        }
        assert!(!machine.step());
        assert_eq!(machine.count_nonblanks(), 11);
        for tape_index in 0..11 {
            assert_eq!(u8::from(machine.tape.read(tape_index)), 35);
        }
    }

    #[test]
    fn bb10_colored_tape_preserves_behavior_and_write_provenance() {
        let mut original = Machine::from_string(BB10).unwrap();
        let rows = original
            .program
            .state_to_symbol_to_action
            .chunks_exact(2)
            .enumerate()
            .map(|(state_index, row)| {
                (0..11)
                    .map(|symbol_index| {
                        let mut action = row[usize::from(symbol_index != 0)].clone();
                        if action.next_symbol != Symbol::STATE_ZERO {
                            action.next_symbol =
                                Symbol::from(u8::try_from(state_index + 1).unwrap());
                        }
                        action
                    })
                    .collect()
            })
            .collect();
        let mut colored = Machine {
            program: Program::from_rows(rows).unwrap(),
            ..Machine::from_string(BB10).unwrap()
        };
        for _step_index in 0..100_000 {
            let writing_state = original.state();
            let tape_index = original.tape_index();
            let continues = original.step();
            assert_eq!(colored.step(), continues);
            assert_eq!(original.state(), colored.state());
            assert_eq!(original.tape_index(), colored.tape_index());
            let expected = if original.tape.read(tape_index) == Symbol::STATE_ZERO {
                0
            } else {
                writing_state + 1
            };
            assert_eq!(u8::from(colored.tape.read(tape_index)), expected);
            if !continues {
                break;
            }
        }
        assert_eq!(original.count_nonblanks(), colored.count_nonblanks());
        let lower = -i64::try_from(original.tape.negative.len()).unwrap();
        let upper = i64::try_from(original.tape.nonnegative.len()).unwrap();
        for tape_index in lower..upper {
            assert_eq!(
                original.tape.read(tape_index) != Symbol::STATE_ZERO,
                colored.tape.read(tape_index) != Symbol::STATE_ZERO
            );
        }
    }

    #[test]
    fn tenth_state_writes_extended_symbol_on_halting_transition() {
        let mut original = Machine::from_string(BB10).unwrap();
        original.state = 9;
        let mut colored = Machine::from_string(&format!("aRZ{}", "0RA".repeat(10))).unwrap();
        assert!(!original.step());
        assert!(!colored.step());
        assert_eq!(u8::from(original.tape.read(0)), 1);
        assert_eq!(u8::from(colored.tape.read(0)), 10);
    }

    #[test]
    fn malformed_extended_tables_are_rejected_before_execution() {
        for program in [
            "aRA",
            "1RA1RA_1RA",
            &"0RA".repeat(37),
            &vec!["0RA"; 27].join("_"),
        ] {
            assert!(Machine::from_string(program).is_err(), "{program}");
        }
    }

    #[test]
    fn undefined_transition_preserves_legacy_reported_halt_state() {
        for state_count in [1, 2, 10, 25] {
            let mut machine = Machine::from_string(&vec!["---"; state_count].join("_")).unwrap();
            assert!(!machine.step());
            assert!(machine.is_halted());
            assert_eq!(machine.state(), 25);
        }
    }

    #[test]
    fn undefined_transition_halts_even_with_a_state_named_z() {
        let mut machine = Machine::from_string(&vec!["---"; 26].join("_")).unwrap();
        assert!(!machine.step());
        assert!(machine.is_halted());
        assert_eq!(machine.state(), u8::MAX);
    }
}
