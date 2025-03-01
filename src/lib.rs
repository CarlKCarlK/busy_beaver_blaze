#![feature(portable_simd)]
use core::simd::prelude::*;
use core::simd::{LaneCount, SupportedLaneCount};
const LANES_CMK: usize = 32;
pub const ALIGN: usize = 64;

// cmk00 Ideas for speedup:
// cmk00    Use nightly simd to average adjacent cells (only useful when at higher smoothing)
// cmk00    Build up 64 (or 128 or 256) rows without merging then use a Rayon parallel tree merge (see https://chatgpt.com/share/67bb94cb-4ba4-800c-b430-c45a5eb46715)
// cmk00    Better than doing a tree merge would be having different final lines processed in parallel

// cmk Could u8 in tape be a bool? (also in average_with_X)

use aligned_vec::AVec;
use arrayvec::ArrayVec;
use core::fmt;
use core::str::FromStr;
use derive_more::Error as DeriveError;
use derive_more::derive::Display;
use instant::Instant;
use itertools::Itertools;
use png::{BitDepth, ColorType, Encoder};
use rayon::{current_num_threads, prelude::*};
use smallvec::SmallVec;
use thousands::Separable;
use wasm_bindgen::prelude::*;

// use web_sys::console;

// cmk is the image size is a power of 2, then don't apply filters (may be a bad idea, because user doesn't control native size exactly)
// cmk0 see if can remove more as_u64()'s

pub const BB2_CHAMP: &str = "
	A	B
0	1RB	1LA
1	1LB	1RH
";

pub const BB3_CHAMP: &str = "
	A	B	C
0	1RB	0RC	1LC
1	1RH	1RB	1LA
";
pub const BB4_CHAMP: &str = "
	A	B	C	D
0	1RB	1LA	1RH	1RD
1	1LB	0LC	1LD	0RA
";

pub const BB5_CHAMP: &str = "
    A	B	C	D	E
0	1RB	1RC	1RD	1LA	1RH
1	1LC	1RB	0LE	1LD	0LA
";

pub const BB6_CONTENDER: &str = "
    	A	B	C	D	E	F
0	1RB	1RC	1LC	0LE	1LF	0RC
1	0LD	0RF	1LA	1RH	0RB	0RE
";

pub const MACHINE_7_135_505_A: &str = "   
0	1
A	1RB	0LD
B	1RC	---
C	1LD	1RA
D	1RE	1LC
E	0LA	0RE
";
pub const MACHINE_7_135_505_B: &str = "1RB0LD_1RC---_1LD1RA_1RE1LC_0LA0RE";

#[derive(Debug)]
struct Tape {
    nonnegative: AVec<u8>,
    negative: AVec<u8>,
}

impl Default for Tape {
    fn default() -> Self {
        Self {
            nonnegative: AVec::new(ALIGN),
            negative: AVec::new(ALIGN),
        }
    }
}

impl Tape {
    //cmki
    #[inline(never)]
    fn read(&self, index: i64) -> u8 {
        if index >= 0 {
            self.nonnegative.get(index as usize).copied().unwrap_or(0)
        } else {
            self.negative
                .get((-index - 1) as usize)
                .copied()
                .unwrap_or(0)
        }
    }

    //cmki
    #[inline(never)]
    #[allow(clippy::shadow_reuse)]
    fn write(&mut self, index: i64, value: u8) {
        let (index, vec) = if index >= 0 {
            (index as usize, &mut self.nonnegative)
        } else {
            ((-index - 1) as usize, &mut self.negative)
        };

        if index == vec.len() {
            // We are exactly one index beyond the current length
            vec.push(value);
        } else {
            // Assert that we're never more than one index beyond
            assert!(
                index < vec.len(),
                "Index is more than one beyond current length!"
            );
            vec[index] = value;
        }
    }
    fn count_ones(&self) -> usize {
        self.nonnegative
            .iter()
            .chain(self.negative.iter()) // Combine both vectors
            .map(|&x| usize::from(x == 1))
            .sum()
    }

    #[cfg(test)]
    #[allow(clippy::min_ident_chars)]
    pub fn index_range_to_string(&self, range: core::ops::RangeInclusive<i64>) -> String {
        let mut s = String::new();
        for i in range {
            s.push_str(&self.read(i).to_string());
        }
        s
    }

    //cmki
    #[inline(never)]
    pub fn min_index(&self) -> i64 {
        -(self.negative.len() as i64)
    }

    //cmki
    #[inline(never)]
    pub fn max_index(&self) -> i64 {
        self.nonnegative.len() as i64 - 1
    }

    //cmki
    #[inline(never)]
    pub fn width(&self) -> u64 {
        (self.max_index() - self.min_index() + 1) as u64
    }
}

#[wasm_bindgen]
pub struct Machine {
    state: u8,
    tape_index: i64,
    tape: Tape,
    program: Program,
}

#[wasm_bindgen]
impl Machine {
    #[wasm_bindgen(constructor)]
    pub fn from_string(input: &str) -> Result<Self, String> {
        input.parse().map_err(|error| format!("{error:?}"))
    }

    #[wasm_bindgen]
    pub fn step(&mut self) -> bool {
        self.next().is_some()
    }

    #[wasm_bindgen]
    //cmki
    #[inline(never)]
    #[must_use]
    pub fn count_ones(&self) -> u32 {
        self.tape.count_ones() as u32
    }

    #[wasm_bindgen]
    //cmki
    #[inline(never)]
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
    type Item = ();

    //cmki
    #[inline(never)]
    fn next(&mut self) -> Option<Self::Item> {
        let program = &self.program;
        let input = self.tape.read(self.tape_index);
        let action = program.action(self.state, input);
        self.tape.write(self.tape_index, action.next_symbol);
        self.tape_index += action.direction as i64;
        self.state = action.next_state;
        if self.state < program.state_count {
            Some(())
        } else {
            None
        }
    }
}

type StateToSymbolToAction = ArrayVec<Action, { Program::MAX_STATE_COUNT }>; // cmk const

#[derive(Debug)]
struct Program {
    state_count: u8,
    // symbol_count: u8,
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
        let is_first_non_space_a_numeral =
            s.trim().chars().next().is_some_and(|c| c.is_ascii_digit());

        match (count_lines, is_first_non_space_a_numeral) {
            (1, _) => Self::parse_standard_format(s),
            (2.., false) => Self::parse_symbol_to_state(s),
            (2.., true) => Self::parse_state_to_symbol(s),
            _ => Err(Error::UnexpectedFormat),
        }
    }
}

impl Program {
    pub const SYMBOL_COUNT: usize = 2;
    pub const MAX_STATE_COUNT: usize = Self::SYMBOL_COUNT * 50;

    //cmki
    #[inline(never)]
    fn action(&self, state: u8, symbol: u8) -> &Action {
        let offset = state as usize * Self::SYMBOL_COUNT + symbol as usize;
        &self.state_to_symbol_to_action[offset]
    }
    fn parse_state(input: impl AsRef<str>) -> Result<char, Error> {
        // println!("cmk {:?}", input.as_ref());
        let mut chars = input.as_ref().chars();
        match (chars.next(), chars.next()) {
            (Some(char @ 'A'..='Z'), None) => Ok(char), // Ensure single uppercase letter
            _ => Err(Error::UnexpectedState),
        }
    }

    #[allow(clippy::shadow_reuse)]
    fn parse_action(part: impl AsRef<str>) -> Result<Action, Error> {
        let part = part.as_ref();
        let asciis = part.as_bytes();
        if asciis.len() != 3 {
            return Err(Error::MissingField);
        }
        if asciis == b"---" {
            return Ok(Action {
                next_state: 25,
                next_symbol: 0,
                direction: -1,
            });
        }
        let next_symbol = match asciis[0] {
            b'0' => 0,
            b'1' => 1,
            _ => return Err(Error::InvalidChar),
        };
        let direction = match asciis[1] {
            b'L' => -1,
            b'R' => 1,
            _ => return Err(Error::InvalidChar),
        };
        let next_state = match asciis[2] {
            b'A'..=b'Z' => asciis[2] - b'A',
            _ => return Err(Error::InvalidChar),
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

        let symbol_count = state_to_symbol_to_action[0].len() as u8;
        if symbol_count == 0 {
            return Err(Error::InvalidSymbolsCount {
                expected: 1,
                got: 0,
            });
        }
        if symbol_count != Self::SYMBOL_COUNT as u8 {
            return Err(Error::InvalidSymbolsCount {
                expected: Self::SYMBOL_COUNT,
                got: symbol_count as usize,
            });
        }

        // Ensure proper dimensions (STATE_COUNT x 2)
        let state_count = state_to_symbol_to_action.len() as u8;
        if state_count == 0 {
            return Err(Error::InvalidStatesCount {
                expected: 1,
                got: 0,
            });
        }

        if state_count > Self::MAX_STATE_COUNT as u8 {
            return Err(Error::InvalidStatesCount {
                expected: Self::MAX_STATE_COUNT,
                got: state_count as usize,
            });
        }

        Ok(Self {
            state_count,
            // symbol_count,
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
        let state_count = state_to_symbol_to_action.len() as u8;
        if state_count == 0 {
            return Err(Error::InvalidStatesCount {
                expected: 1,
                got: 0,
            });
        }

        if state_count > Self::MAX_STATE_COUNT as u8 {
            return Err(Error::InvalidStatesCount {
                expected: Self::MAX_STATE_COUNT,
                got: state_count as usize,
            });
        }

        let symbol_count = state_to_symbol_to_action[0].len() as u8;
        if symbol_count == 0 {
            return Err(Error::InvalidSymbolsCount {
                expected: 1,
                got: 0,
            });
        }

        if symbol_count != Self::SYMBOL_COUNT as u8 {
            return Err(Error::InvalidSymbolsCount {
                expected: Self::SYMBOL_COUNT,
                got: symbol_count as usize,
            });
        }

        Ok(Self {
            state_count,
            // symbol_count,
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
            return Err(Error::InvalidSymbolsCount {
                expected: 1,
                got: 0,
            });
        }

        if symbol_count != Self::SYMBOL_COUNT {
            return Err(Error::InvalidSymbolsCount {
                expected: Self::SYMBOL_COUNT,
                got: symbol_count,
            });
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
            state_count: state_count as u8,
            // symbol_count: symbol_count as u8,
            state_to_symbol_to_action: state_to_symbol_to_action.into_iter().flatten().collect(),
        })
    }
}

#[derive(Debug)]
struct Action {
    next_state: u8,
    next_symbol: u8,
    direction: i8,
}

/// A trait for iterators that can print debug output at intervals.
pub trait DebuggableIterator: Iterator {
    /// Runs the iterator while printing debug output at intervals.
    //cmki
    #[inline(never)]
    fn debug_count(&mut self, debug_interval: usize) -> usize
    where
        Self: Sized + core::fmt::Debug, // ✅ Ensure Debug is implemented
    {
        let mut step_index = 0;
        let mut countdown = debug_interval; // New countdown variable

        println!("Step {}: {:?}", step_index.separate_with_commas(), self);

        while self.next().is_some() {
            step_index += 1;
            countdown -= 1;

            if countdown == 0 {
                println!("Step {}: {:?}", step_index.separate_with_commas(), self);
                countdown = debug_interval; // Reset countdown
            }
        }

        step_index + 1 // Convert last index into count
    }
}

// Implement the trait for all Iterators
#[allow(clippy::missing_trait_methods)]
impl<T> DebuggableIterator for T where T: Iterator + core::fmt::Debug {}

/// Error type for parsing a `Program` from a string.
#[derive(Debug, Display, DeriveError)]
pub enum Error {
    #[display("Invalid number format: {}", _0)]
    ParseIntError(core::num::ParseIntError),

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

    #[display("Unexpected state encountered")]
    UnexpectedState,

    #[display("Invalid encoding encountered")]
    EncodingError,

    #[display("Unexpected format")]
    UnexpectedFormat,
}

// Implement conversions manually where needed
impl From<core::num::ParseIntError> for Error {
    fn from(err: core::num::ParseIntError) -> Self {
        Self::ParseIntError(err)
    }
}
#[repr(transparent)]
#[derive(Debug, Default, Display, Copy, Clone)]
struct Pixel(u8);

impl Pixel {
    const WHITE: Self = Self(0);
    const SPLAT_1: Simd<u8, LANES_CMK> = Simd::<u8, LANES_CMK>::splat(1);
    //cmki
    #[inline(never)]
    fn slice_merge_with_white(pixels: &mut [Self]) {
        // Safety: Pixel is repr(transparent) around u8, so this cast is safe
        let bytes: &mut [u8] = unsafe {
            core::slice::from_raw_parts_mut(pixels.as_mut_ptr().cast::<u8>(), pixels.len())
        };
        // cmk00000 Look at zerocopy or bytemuck

        // Process with SIMD where possible
        let (prefix, chunks, suffix) = bytes.as_simd_mut::<LANES_CMK>();

        // Process SIMD chunks
        for chunk in chunks {
            *chunk >>= Self::SPLAT_1;
        }

        // Process remaining elements
        for byte in prefix.iter_mut().chain(suffix.iter_mut()) {
            *byte >>= 1;
        }
    }

    //cmki
    #[inline(never)]
    fn slice_merge(left: &mut [Self], right: &[Self]) {
        //     for (left_pixel, right_pixel) in left.iter_mut().zip(right.iter()) {
        //         left_pixel.merge(*right_pixel);
        //     }
        // }

        debug_assert_eq!(
            left.len(),
            right.len(),
            "Both slices must have the same length"
        );

        // Safety: Pixel is repr(transparent) around u8, so this cast is safe //cmk000 look at zerocopy or bytemuck
        let left_bytes: &mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(left.as_mut_ptr().cast::<u8>(), left.len()) };

        let right_bytes: &[u8] =
            unsafe { core::slice::from_raw_parts(right.as_ptr().cast::<u8>(), right.len()) };

        // Process chunks with SIMD where possible
        let (left_prefix, left_chunks, left_suffix) = left_bytes.as_simd_mut::<LANES_CMK>();
        let (right_prefix, right_chunks, right_suffix) = right_bytes.as_simd::<LANES_CMK>();

        // Process SIMD chunks using (a & b) + ((a ^ b) >> 1) formula
        for (left_chunk, right_chunk) in left_chunks.iter_mut().zip(right_chunks.iter()) {
            let a_and_b = *left_chunk & *right_chunk;
            *left_chunk ^= *right_chunk;
            *left_chunk >>= Self::SPLAT_1;
            *left_chunk += a_and_b;
        }

        // println!("cmk len of total: {}", left.len());
        // println!("cmk len of prefix: {}", left_prefix.len());
        // println!("cmk len of suffix: {}", left_suffix.len());
        assert!(left_prefix.is_empty() && right_prefix.is_empty());
        // assert!(left_suffix.is_empty() && right_suffix.is_empty());
        // Process remaining elements in prefix
        // for (left_byte, right_byte) in left_prefix.iter_mut().zip(right_prefix.iter()) {
        //     *left_byte = (*left_byte & *right_byte) + ((*left_byte ^ *right_byte) >> 1);
        // }

        // Process remaining elements in suffix
        for (left_byte, right_byte) in left_suffix.iter_mut().zip(right_suffix.iter()) {
            *left_byte = (*left_byte & *right_byte) + ((*left_byte ^ *right_byte) >> 1);
        }
    }
    //cmki
    #[inline(never)]
    const fn merge(&mut self, other: Self) {
        self.0 = (self.0 >> 1) + (other.0 >> 1) + ((self.0 & other.0) & 1);
    }

    //cmki
    #[inline(never)]
    fn merge_slice_down_sample(
        slice: &[Self],
        empty_count: usize,
        down_step: PowerOfTwo,
        down_step_usize: usize,
    ) -> Self {
        debug_assert!(down_step.as_usize() == down_step_usize);
        let mut sum: usize = 0;
        for i in (0..slice.len()).step_by(down_step_usize) {
            sum += slice[i].0 as usize;
        }
        let total_len = PowerOfTwo::from_usize(slice.len() + empty_count);
        let count = total_len.saturating_div(down_step);
        let mean = count.divide_into(sum) as u8;
        Self(mean)
    }

    //cmki
    #[inline(never)]
    #[inline(never)]
    fn merge_slice_all(slice: &[Self], empty_count: i64) -> Self {
        let sum: u32 = slice.iter().map(|pixel: &Self| pixel.0 as u32).sum();
        let count = slice.len() + empty_count as usize;
        debug_assert!(count.is_power_of_two(), "Count must be a power of two");
        Self(PowerOfTwo::from_u64(count as u64).divide_into(sum) as u8)
    }
}

impl From<u8> for Pixel {
    //cmki
    #[inline(never)]
    fn from(value: u8) -> Self {
        debug_assert!(value <= 1, "Input value must be 0 or 1, got {value}");
        Self(value * 255)
    }
}

#[derive(Clone, Debug)]
struct Spaceline {
    sample: PowerOfTwo,
    negative: AVec<Pixel>,
    nonnegative: AVec<Pixel>,
    time: u64,
    smoothness: PowerOfTwo,
}

// impl core::ops::Index<i64> for Spaceline {
//     type Output = Pixel;

//     fn index(&self, index: i64) -> &Self::Output {
//         if index < 0 {
//             self.negative
//                 .get((-index - 1) as usize)
//                 .unwrap_or(&Pixel::WHITE)
//         } else {
//             self.nonnegative
//                 .get(index as usize)
//                 .unwrap_or(&Pixel::WHITE)
//         }s
//     }
// }

impl Spaceline {
    // cmk good name??
    fn new0(smoothness: PowerOfTwo) -> Self {
        let mut v = AVec::new(1);
        v.push(Pixel::WHITE);

        Self {
            sample: PowerOfTwo::ONE,
            negative: AVec::new(ALIGN),
            nonnegative: v,
            time: 0,
            smoothness,
        }
    }

    // cmk0000 should remove this function
    //cmki
    #[inline(never)]
    fn pixel_index(&self, index: usize) -> Pixel {
        let negative_len = self.negative.len();
        if index < negative_len {
            self.negative[negative_len - 1 - index]
        } else {
            self.nonnegative[index - negative_len]
        }
    }

    // cmk0000 should remove this function
    //cmki
    #[inline(never)]
    fn pixel_index_unbounded(&self, index: usize) -> Pixel {
        let negative_len = self.negative.len();
        if index < negative_len {
            self.negative
                .get(negative_len - 1 - index)
                .copied()
                .unwrap_or_default()
        } else {
            self.nonnegative
                .get(index - negative_len)
                .copied()
                .unwrap_or_default()
        }
    }

    // cmk0000000000 must remove this function
    //cmki
    #[inline(never)]
    #[inline(never)]
    fn new2(
        sample: PowerOfTwo,
        start: i64,
        pixels: AVec<Pixel>,
        time: u64,
        smoothness: PowerOfTwo,
    ) -> Self {
        let mut result = Self {
            sample,
            negative: AVec::new(ALIGN),
            nonnegative: pixels,
            time,
            smoothness,
        };
        while result.tape_start() > start {
            result.negative.insert(0, result.nonnegative.remove(0));
        }
        result
    }

    // cmk0000000000 must remove this function
    fn pixel_range(&self, start: usize, end: usize) -> Vec<Pixel> {
        // Ensure start and end are within valid bounds
        assert!(
            start <= end,
            "start index {start} must be <= end index {end}"
        );

        // Create a vector with enough capacity
        let mut result = Vec::with_capacity(end - start);

        // Simply loop through all indices and call pixel_index
        for i in start..end {
            result.push(self.pixel_index(i));
        }

        result
    }

    // cmk0000 should remove this function
    //cmki
    #[inline(never)]
    fn pixel_index_mut(&mut self, index: usize) -> &mut Pixel {
        let negative_len = self.negative.len();
        if index < negative_len {
            &mut self.negative[negative_len - 1 - index]
        } else {
            &mut self.nonnegative[index - negative_len]
        }
    }

    //cmki
    #[inline(never)]
    fn tape_start(&self) -> i64 {
        -((self.sample * self.negative.len()) as i64)
    }

    //cmki
    #[inline(never)]
    fn len(&self) -> usize {
        self.nonnegative.len() + self.negative.len()
    }

    //cmki
    #[inline(never)]
    fn resample_if_needed(&mut self, sample: PowerOfTwo) {
        // Sampling & Averaging 2 --
        // When we merge rows, we sometimes need to squeeze the earlier row to
        // have the same sample rate as the later row.

        assert!(!self.nonnegative.is_empty(), "real assert a");
        assert!(
            self.sample.divides_i64(self.tape_start()),
            "Start must be a multiple of the sample rate",
        );
        if sample == self.sample {
            return;
        }
        let cells_to_add = sample.rem_euclid_into(self.tape_start());
        let new_tape_start = self.tape_start() - cells_to_add;
        let old_items_to_add = self.sample.divide_into(cells_to_add);
        let old_items_per_new = sample / self.sample;
        let old_items_per_new_u64 = old_items_per_new.as_u64();
        let old_items_per_new_usize = old_items_per_new_u64 as usize;

        assert!(sample >= self.sample, "real assert 12");
        let old_items_to_use = old_items_per_new.as_u64() - old_items_to_add as u64;
        assert!(old_items_to_use <= self.len() as u64, "real assert d10");

        let down_step = sample.saturating_div(self.smoothness);
        let pixel0 = Pixel::merge_slice_down_sample(
            &self.pixel_range(0, old_items_to_use as usize),
            old_items_to_add as usize,
            down_step,
            down_step.as_usize(),
        );

        let mut new_index = 0usize;
        *self.pixel_index_mut(new_index) = pixel0;
        new_index += 1;
        let value_len = self.len() as u64;

        let down_size_usize = down_step.as_usize();
        for old_index in (old_items_to_use..value_len).step_by(old_items_per_new_usize) {
            let old_end = (old_index + old_items_to_use).min(value_len);
            let slice = &self.pixel_range(old_index as usize, old_end as usize);
            let old_items_to_add_inner = old_items_per_new_u64 - (old_end - old_index);
            *self.pixel_index_mut(new_index) = Pixel::merge_slice_down_sample(
                slice,
                old_items_to_add_inner as usize,
                down_step,
                down_size_usize,
            );
            new_index += 1;
        }

        // trim the vector to the new length
        self.pixel_restart(new_tape_start, new_index, sample);
    }

    // fn pixel_restart0(&mut self, tape_start: i64, len: usize) {
    //     assert!(self.tape_start() <= tape_start, "real assert 11");
    //     while self.tape_start() < tape_start {
    //         self.nonnegative.insert(0, self.negative.remove(0));
    //     }
    //     assert!(self.len() >= len, "real assert 12");
    //     while self.len() > len {
    //         self.nonnegative.pop();
    //     }
    //     assert!(self.len() == len, "real assert 13");
    // }

    #[inline(never)]
    fn pixel_restart(&mut self, tape_start: i64, len: usize, sample: PowerOfTwo) {
        self.sample = sample;
        assert!(self.tape_start() <= tape_start, "real assert 11");
        while self.tape_start() < tape_start {
            self.nonnegative.insert(0, self.negative.remove(0));
        }
        assert!(self.len() >= len, "real assert 12");
        while self.len() > len {
            self.nonnegative.pop();
        }
        assert!(self.len() == len, "real assert 13");
    }

    // cmk00000 simd?
    //cmki
    #[inline(never)]
    fn resample_if_needed_slower(&mut self, sample: PowerOfTwo) {
        // Sampling & Averaging 2 --
        // When we merge rows, we sometimes need to squeeze the earlier row to
        // have the same sample rate as the later row.

        if sample == self.sample {
            return;
        }
        assert!(!self.nonnegative.is_empty(), "real assert a");
        assert!(
            self.sample.divides_i64(self.tape_start()),
            "Start must be a multiple of the sample rate",
        );
        //        let previous_tape_length = self.sample * self.len();

        assert!(sample > self.sample, "real assert 12");
        // e.g. sample was 2 and now it will be 8, so step is 4.
        let step = sample / self.sample;
        // if self.smoothness == 1, we always just take the first pixel so substep is 4
        // if self.smoothness == 2 and our sample is 2, we still just take the first pixel, substep is 4
        // if self.smoothness == 4 and our sample is 2, then substep will be half of 4, so substep is 2.
        // if self.smoothness == 8 and our sample is 2, then substep will be 1 and we'll average everything, substep is 1.
        let substep = sample.saturating_div(self.smoothness).min(step);
        // if substep == PowerOfTwo::ONE {
        // println!(
        //     "cmk old sample {:?}, new {sample:?}, step: {step:?}, substep: {substep:?}, len: {}",
        //     self.sample,
        //     self.len()
        // );
        // }

        for pixels in [&mut self.negative, &mut self.nonnegative] {
            // e.g. samples was 2 and we had 5 values (so tape length of 10), now samples will be 8 and we'll have one 2 value (so tape length of 16)
            // more over, we might just put one old value into the new, average every 2nd or 4th, etc, or average all.
            // So, we need a step and substep. If they are equal, we only take the first pixel. If substep is 1, we average everything.
            // otherwise, we add up and average.
            // Also, on the last index position, we may may be short and need to add in virtual pixels.
            let pixels_len = pixels.len();
            let mut pixel;
            let mut new_index = 0;
            for old_index in (0..pixels.len()).step_by(step.as_usize()) {
                if substep == step {
                    pixel = pixels[old_index];
                }
                // } else if substep == PowerOfTwo::ONE {
                //     let needed_fill_ins = step.offset_to_align(pixels_len);
                //     pixel = Pixel::merge_slice_all(
                //         &pixels[old_index..old_index + step.as_usize() - needed_fill_ins],
                //         needed_fill_ins as i64,
                //     );
                else {
                    let mut sum: u32 = 0;
                    for sub_index in (0..step.as_usize()).step_by(substep.as_usize()) {
                        let index2 = old_index + sub_index;
                        if index2 >= pixels_len {
                            break;
                        }
                        sum += pixels[index2].0 as u32;
                    }
                    pixel = Pixel(step.divide_into(sum) as u8);
                }
                pixels[new_index] = pixel;
                new_index += 1;
            }
            // trim the vector to the new length
            pixels.truncate(new_index);
        }

        self.sample = sample;
    }

    //cmki
    #[inline(never)]
    fn merge(&mut self, other: &Self) {
        // cmk change to debug_assert?
        assert!(self.time < other.time, "real assert 2");
        assert!(self.sample <= other.sample, "real assert 3");
        assert!(self.tape_start() >= other.tape_start(), "real assert 4");
        assert!(self.tape_start() >= other.tape_start(), "real assert 6b");
        self.resample_if_needed(other.sample);
        assert!(self.sample == other.sample, "real assert 5b");
        assert!(self.tape_start() >= other.tape_start(), "real assert 6c");

        // cmk000 could be done in one step
        while self.tape_start() > other.tape_start() {
            self.negative.push(Pixel::WHITE);
        }
        assert!(self.tape_start() == other.tape_start(), "real assert 6c");

        // cmk000 could be done in one step
        while self.len() < other.len() {
            self.nonnegative.push(Pixel::WHITE);
        }
        assert!(self.len() == other.len(), "real assert 6d");

        // // align
        // // cmk000 replace Eight
        // let needed_padding = PowerOfTwo::from_usize_const(LANES)
        //     // .saturating_div(PowerOfTwo::FOUR)
        //     .offset_to_align(self.nonnegative.len());
        // for _ in 0..needed_padding {
        //     self.nonnegative.push(Pixel::WHITE);
        //     other.nonnegative.push(Pixel::WHITE);
        // }

        Pixel::slice_merge(&mut self.nonnegative, &other.nonnegative);
    }

    //cmki
    #[inline(never)]
    #[allow(clippy::integer_division_remainder_used)]
    fn new(tape: &Tape, x_goal: u32, step_index: u64, x_smoothness: PowerOfTwo) -> Self {
        // Sampling & Averaging 4 --
        let tape_width = tape.width();
        let tape_min_index = tape.min_index();
        let tape_max_index = tape.max_index();
        let x_sample = sample_rate(tape_width, x_goal);

        if step_index % 10_000_000 == 0 {
            println!(
                "cmk Spaceline::new step_index {}, tape width {tape_width:?} ({tape_min_index}..={tape_max_index}), x_sample {:?}, x_goal {x_goal:?}",
                step_index.separate_with_commas(),
                x_sample.as_usize()
            );
        }

        // cmk0000000000000
        // cmk00000 would make a special case for x_sample=1 (just copy * 255) and then the averages wouldn't need to check it
        if x_smoothness >= x_sample {
            let (negative, nonnegative) = match x_sample {
                PowerOfTwo::ONE | PowerOfTwo::TWO | PowerOfTwo::FOUR => (
                    average_with_iterators(&tape.negative, x_sample),
                    average_with_iterators(&tape.nonnegative, x_sample),
                ),
                // PowerOfTwo::FOUR => (
                //     average_with_simd::<4>(&tape.negative, x_sample),
                //     average_with_simd::<4>(&tape.nonnegative, x_sample),
                // ),
                PowerOfTwo::EIGHT => (
                    average_with_simd::<8>(&tape.negative, x_sample),
                    average_with_simd::<8>(&tape.nonnegative, x_sample),
                ),
                PowerOfTwo::SIXTEEN => (
                    average_with_simd::<16>(&tape.negative, x_sample),
                    average_with_simd::<16>(&tape.nonnegative, x_sample),
                ),
                PowerOfTwo::THIRTY_TWO => (
                    // _ => (
                    average_with_simd::<32>(&tape.negative, x_sample),
                    average_with_simd::<32>(&tape.nonnegative, x_sample),
                ),
                _ => (
                    average_with_simd::<64>(&tape.negative, x_sample),
                    average_with_simd::<64>(&tape.nonnegative, x_sample),
                ),
            };
            // cmk00000000
            // do an unsafe cast from from u8 to Pixel
            let negative: AVec<Pixel> = unsafe { core::mem::transmute(negative) };
            let nonnegative: AVec<Pixel> = unsafe { core::mem::transmute(nonnegative) };

            return Self {
                sample: x_sample,
                negative,
                nonnegative,
                time: step_index,
                smoothness: x_smoothness,
            };
        }

        let sample_start: i64 = tape_min_index - x_sample.rem_euclid_into(tape_min_index);
        assert!(
            sample_start <= tape_min_index
                && x_sample.divides_i64(sample_start)
                && tape_min_index - sample_start < x_sample.as_u64() as i64,
            "real assert b1"
        );

        let mut pixels = AVec::with_capacity(ALIGN, x_goal as usize * 2); // cmk000 const

        let down_sample = x_sample.min(x_smoothness);
        let down_step = x_sample.saturating_div(down_sample);

        if down_sample == PowerOfTwo::ONE {
            // With least smoothness, we just read the pixels directly.
            for sample_index in (sample_start..=tape_max_index).step_by(x_sample.as_usize()) {
                // Directly read and convert the pixel from the tape.
                pixels.push(tape.read(sample_index).into());
            }
        } else {
            // For x_sample > 1, process as before.

            // Create a temporary vector to hold x_sample pixels.
            let mut pixel_range: SmallVec<[Pixel; 64]> =
                SmallVec::from_elem(Pixel(0), down_sample.as_usize());
            for sample_index in (sample_start..=tape_max_index).step_by(x_sample.as_usize()) {
                for (i, pixel) in pixel_range.iter_mut().enumerate() {
                    *pixel = tape.read(sample_index + (down_step * i) as i64).into();
                }
                let pixel = Pixel::merge_slice_all(&pixel_range, 0);
                pixels.push(pixel);
            }
        }

        Self::new2(x_sample, sample_start, pixels, step_index, x_smoothness)
    }
}

struct Spacelines {
    main: Vec<Spaceline>,
    buffer0: Vec<(Spaceline, PowerOfTwo)>, // cmk0 better names
    buffer1: Vec<Option<(u64, Spaceline)>>,
    buffer1_capacity: PowerOfTwo,
}

impl Spacelines {
    fn new(smoothness: PowerOfTwo, buffer1_count: PowerOfTwo) -> Self {
        Self {
            main: vec![Spaceline::new0(smoothness)],
            buffer0: Vec::new(),
            buffer1: Vec::with_capacity(buffer1_count.as_usize()),
            buffer1_capacity: buffer1_count,
        }
    }
    fn len(&self) -> usize {
        self.main.len() + usize::from(!self.buffer0.is_empty())
    }

    fn get<'a>(&'a self, index: usize, last: &'a Spaceline) -> &'a Spaceline {
        if index == self.len() - 1 {
            last
        } else {
            &self.main[index]
        }
    }

    fn flush_buffer0(&mut self) {
        self.flush_buffer1();
        // We now have a buffer that needs to be flushed at the end
        if !self.buffer0.is_empty() {
            assert!(self.buffer0.len() == 1, "real assert 13");
            self.main.push(self.buffer0.pop().unwrap().0);
        }
    }

    #[allow(clippy::min_ident_chars)]
    //cmki
    #[inline(never)]
    fn compress_average(&mut self) {
        assert!(self.buffer0.is_empty(), "real assert b2");
        assert!(fast_is_even(self.main.len()), "real assert 11");
        // println!("cmk compress_average");

        self.main = self
            .main
            .drain(..)
            .tuples()
            .map(|(mut a, b)| {
                assert!(a.tape_start() >= b.tape_start(), "real assert 4a");
                a.merge(&b);
                a
            })
            .collect();
    }

    //cmki
    #[inline(never)]
    fn compress_take_first(&mut self, new_sample: PowerOfTwo) {
        assert!(self.buffer0.is_empty(), "real assert e2");
        assert!(fast_is_even(self.main.len()), "real assert e11");
        // println!("cmk compress_take_first");
        self.main
            .retain(|spaceline| new_sample.divides_u64(spaceline.time));
    }

    fn flush_buffer1(&mut self) {
        if self.buffer1.is_empty() {
            return;
        }
        let mut whole = core::mem::take(&mut self.buffer1);
        let mut start = 0usize;
        while start < whole.len() {
            let end = start + prev_power_of_two(whole.len() - start);
            let slice = &mut whole[start..end];
            debug_assert!(slice.len().is_power_of_two(), "real assert 10");
            let slice_len = PowerOfTwo::from_usize(slice.len());
            let weight = PowerOfTwo::from_usize(slice.len());

            // Binary tree reduction algorithm
            let mut gap = PowerOfTwo::ONE;

            while gap.as_usize() < slice.len() {
                let pair_count = slice_len.saturating_div(gap);
                // cmk0000 for now, always run sequentially
                if pair_count >= PowerOfTwo::MAX {
                    // Process pairs in parallel
                    slice
                        .par_chunks_mut(gap.double().as_usize())
                        .for_each(|chunk| {
                            let (left_index, right_index) = (0, gap.as_usize());
                            let (_, right_spaceline) = chunk[right_index].take().unwrap();
                            let (_, left_spaceline) = chunk[left_index].as_mut().unwrap();
                            left_spaceline.merge(&right_spaceline);
                        });
                } else {
                    slice.chunks_mut(gap.double().as_usize()).for_each(|chunk| {
                        let (left_index, right_index) = (0, gap.as_usize());
                        let (_, right_spaceline) = chunk[right_index].take().unwrap();
                        let (_, left_spaceline) = chunk[left_index].as_mut().unwrap();
                        left_spaceline.merge(&right_spaceline);
                    });
                }
                gap = gap.double();
            }

            // Extract the final merged result
            let (first_index, merged_spaceline) = slice[0].take().unwrap();
            Self::push_internal(&mut self.buffer0, first_index, merged_spaceline, weight);
            start = end;
        }
    }

    fn last(
        &mut self,
        step_index: u64,
        y_sample: PowerOfTwo,
        y_smoothness: PowerOfTwo,
    ) -> Spaceline {
        // If we're going to create a PNG, we need to move buffer1 into buffer0
        self.flush_buffer1();

        if self.buffer0.is_empty() {
            // cmk would be nice to remove this clone
            return self.main.last().unwrap().clone();
        }
        // cmk in the special case in which the sample is 1 and the buffer is 1, can't we just return the buffer's item (as a ref???)

        let buffer_last = self.buffer0.last().unwrap();
        let spaceline_last = &buffer_last.0;
        let weight = buffer_last.1; // cmk0000 should this be used?
        let time = spaceline_last.time;
        let start = spaceline_last.tape_start();
        let x_sample = spaceline_last.sample;
        let last_inside_index = y_sample.rem_into_u64(step_index);

        // cmk we have to clone because we compress in place (clone only half???)
        let mut buffer0 = self.buffer0.clone();
        for inside_index in last_inside_index + 1..y_sample.as_u64() {
            let down_step = y_sample.saturating_div(y_smoothness);
            if !down_step.divides_u64(inside_index) {
                continue;
            }
            let inside_inside_index = down_step.divide_into(inside_index);

            let empty_pixels = AVec::from_iter(
                ALIGN,
                core::iter::repeat_n(Pixel::WHITE, spaceline_last.len()),
            );
            let empty = Spaceline::new2(
                x_sample,
                start,
                empty_pixels,
                time + inside_index - last_inside_index,
                spaceline_last.smoothness,
            );
            Self::push_internal(&mut buffer0, inside_inside_index, empty, PowerOfTwo::ONE);
        }
        assert!(buffer0.len() == 1, "real assert b3");
        buffer0.pop().unwrap().0
    }

    //cmki
    #[inline(never)]
    fn push_internal(
        buffer0: &mut Vec<(Spaceline, PowerOfTwo)>,
        _inside_index: u64,
        mut spaceline: Spaceline,
        mut weight: PowerOfTwo,
    ) {
        while let Some((_last_mut_powerline, last_mut_weight)) = buffer0.last_mut() {
            // If current weight is smaller, just append to buffer
            if weight < *last_mut_weight {
                buffer0.push((spaceline, weight));
                return;
            }

            debug_assert!(
                weight == *last_mut_weight,
                "Weight equality invariant violation"
            );

            // Get ownership of the last element by popping
            let (mut last_powerline, last_weight) = buffer0.pop().unwrap();

            // Merge spacelines and double weight
            last_powerline.merge(&spaceline);

            // Continue with the merged spaceline and doubled weight
            spaceline = last_powerline;
            weight = last_weight.double();

            // If buffer is now empty, push and return
            if buffer0.is_empty() {
                buffer0.push((spaceline, weight));
                return;
            }
        }

        // Handle empty buffer case
        buffer0.push((spaceline, weight));
    }

    //cmki
    #[inline(never)]
    fn push(&mut self, inside_index: u64, y_sample: PowerOfTwo, spaceline: Spaceline) {
        // Calculate buffer capacity
        let capacity = self.buffer1_capacity.min(y_sample);

        // If less than 4 spacelines go into one image line, skip buffer1
        if capacity < PowerOfTwo::FOUR {
            Self::push_internal(&mut self.buffer0, inside_index, spaceline, PowerOfTwo::ONE);
        } else if self.buffer1.len() < capacity.as_usize() {
            self.buffer1.push(Some((inside_index, spaceline)));
        } else {
            self.flush_buffer1();
            self.buffer1.push(Some((inside_index, spaceline)));
        }
    }
}

pub struct SampledSpaceTime {
    step_index: u64,
    x_goal: u32,
    y_goal: u32,
    sample: PowerOfTwo,
    spacelines: Spacelines,
    x_smoothness: PowerOfTwo,
    y_smoothness: PowerOfTwo,
}

/// Create a new in which you give the `x_goal` (space)
/// and the `y_goal` (time). The sample starts at 1 and
/// inner is a vector of one spaceline
impl SampledSpaceTime {
    //cmki
    #[inline(never)]
    #[must_use]
    pub fn new(
        x_goal: u32,
        y_goal: u32,
        x_smoothness: PowerOfTwo,
        y_smoothness: PowerOfTwo,
        buffer1_count: PowerOfTwo,
    ) -> Self {
        Self {
            step_index: 0,
            x_goal,
            y_goal,
            sample: PowerOfTwo::ONE,
            spacelines: Spacelines::new(x_smoothness, buffer1_count),
            x_smoothness,
            y_smoothness,
        }
    }

    fn compress_if_needed(&mut self) {
        // Sampling & Averaging 1--
        // We sometimes need to squeeze rows by averaging adjacent pairs of rows.
        // The alternative is just to keep the 1st row and discard the 2nd row.

        let new_sample = sample_rate(self.step_index, self.y_goal);
        if new_sample != self.sample {
            assert!(
                new_sample / self.sample == PowerOfTwo::TWO,
                "real assert 10"
            );
            self.sample = new_sample;
            if new_sample <= self.y_smoothness {
                self.spacelines.compress_average();
            } else {
                self.spacelines.compress_take_first(new_sample);
            }
        }
    }

    // ideas
    // use
    //       assert!(self.sample.is_power_of_two(), "Sample must be a power of two");
    //       // Use bitwise AND for fast divisibility check
    //       if self.step_index & (self.sample - 1) != 0 {
    //  Also: Inline the top part of the function.
    //  Maybe pre-subtract 1 from sample

    //cmki
    #[inline(never)]
    fn snapshot(&mut self, machine: &Machine) {
        self.step_index += 1;
        let inside_index = self.sample.rem_into_u64(self.step_index);

        // if inside_index == self.sample.as_u64() - 1 {
        //     println!(
        //         "cmk snapshot {} {:?} {inside_index} buf: 0:{} 1:{}/{:?}",
        //         self.step_index,
        //         self.sample,
        //         self.spacelines.buffer0.len(),
        //         self.spacelines.buffer1.len(),
        //         self.spacelines.buffer1_capacity
        //     );
        // }

        if inside_index == 0 {
            // We're starting a new set of spacelines, so flush the buffer and compress (if needed)
            self.spacelines.flush_buffer0();
            self.compress_if_needed();
        }

        let down_step = self.sample.saturating_div(self.y_smoothness);
        if !down_step.divides_u64(inside_index) {
            return;
        }
        let inside_inside_index = down_step.divide_into(inside_index);
        let spaceline = Spaceline::new(
            &machine.tape,
            self.x_goal,
            self.step_index,
            self.x_smoothness,
        );

        // if inside_index == self.sample.as_u64() - 1 {
        //     println!(
        //         "   cmk snapshot {} {:?} {inside_index} buf: 0:{} 1:{}/{:?} {inside_inside_index}",
        //         self.step_index,
        //         self.sample,
        //         self.spacelines.buffer0.len(),
        //         self.spacelines.buffer1.len(),
        //         self.spacelines.buffer1_capacity
        //     );
        // }

        self.spacelines
            .push(inside_inside_index, self.sample, spaceline);
    }

    #[allow(clippy::wrong_self_convention)] // cmk00 consider better name to this function
    fn to_png(&mut self) -> Result<Vec<u8>, Error> {
        let last = self
            .spacelines
            .last(self.step_index, self.sample, self.y_smoothness);
        let x_sample: PowerOfTwo = last.sample;
        let tape_width: u64 = (x_sample * last.len()) as u64;
        let tape_min_index = last.tape_start();
        let x_actual: u32 = x_sample.divide_into(tape_width) as u32;
        let y_actual: u32 = self.spacelines.len() as u32;

        let row_bytes = x_actual;
        let mut packed_data = vec![0u8; row_bytes as usize * y_actual as usize];

        for y in 0..y_actual {
            let spaceline = self.spacelines.get(y as usize, &last);
            let local_start = &spaceline.tape_start();
            let local_x_sample = spaceline.sample;
            let local_per_x_sample = x_sample / local_x_sample;
            let row_start_byte_index: u32 = y * row_bytes;
            let x_start = x_sample.div_ceil_into(local_start - tape_min_index);
            for x in x_start as u32..x_actual {
                let tape_index: i64 = (x_sample * x as usize) as i64 + tape_min_index;
                // cmk consider changing asserts to debug_asserts
                assert!(
                    tape_index >= *local_start,
                    "real assert if x_start is correct"
                );

                let local_spaceline_start: i64 =
                    local_x_sample.divide_into(tape_index - local_start);

                // this helps medium bb6 go from 5 seconds to 3.5
                if local_per_x_sample == PowerOfTwo::ONE || self.x_smoothness == PowerOfTwo::ONE {
                    {
                        if local_spaceline_start >= spaceline.len() as i64 {
                            break;
                        }
                    }
                    let pixel = spaceline
                        .pixel_index_unbounded(local_spaceline_start as usize)
                        .0;
                    if pixel != 0 {
                        let byte_index: u32 = x + row_start_byte_index;
                        packed_data[byte_index as usize] = pixel;
                    }
                    continue;
                }
                // cmk LATER can we make this after by precomputing the collect outside the loop?
                let slice = (local_spaceline_start
                    ..local_spaceline_start + local_per_x_sample.as_u64() as i64)
                    .map(|i| spaceline.pixel_index_unbounded(i as usize))
                    .collect::<Vec<_>>();
                // cmk LATER look at putting this back in
                // if local_spaceline_index >= spaceline.pixels.len() as i64 {
                //     break;
                // }

                // Sample & Averaging 5 --
                let pixel = Pixel::merge_slice_all(&slice, 0).0;
                if pixel != 0 {
                    let byte_index: u32 = x + row_start_byte_index;
                    packed_data[byte_index as usize] = pixel;
                }
            }
        }

        encode_png(x_actual, y_actual, &packed_data)
    }
}

fn sample_rate(row: u64, goal: u32) -> PowerOfTwo {
    let threshold = 2 * goal;
    let ratio = (row + 1) as f64 / threshold as f64;

    // For rows < threshold, ratio < 1, log2(ratio) is negative, and the ceil clamps to 0.
    let exponent = ratio.log2().ceil().max(0.0) as u8;
    PowerOfTwo::from_exp(exponent)
}

#[allow(clippy::integer_division_remainder_used)]
fn encode_png(width: u32, height: u32, image_data: &[u8]) -> Result<Vec<u8>, Error> {
    let mut buf = Vec::new();
    {
        if image_data.len() != (width * height) as usize {
            return Err(Error::EncodingError);
        }
        let mut encoder = Encoder::new(&mut buf, width, height);
        encoder.set_color(ColorType::Indexed);
        encoder.set_depth(BitDepth::Eight);

        // Generate a palette with 256 shades from white (255,255,255) to bright orange (255,165,0)
        let mut palette = Vec::with_capacity(256 * 3);
        for i in 0u16..256 {
            let green = 255 - ((255 - 165) * i / 255); // Green fades from 255 to 165
            let blue = 255 - (255 * i / 255); // Blue fades from 255 to 0
            palette.extend_from_slice(&[255, green as u8, blue as u8]);
        }

        // Set the palette before writing the header
        encoder.set_palette(palette);

        let mut writer = encoder.write_header().map_err(|_| Error::EncodingError)?;
        writer
            .write_image_data(image_data)
            .map_err(|_| Error::EncodingError)?;
    };
    Ok(buf)
}

#[wasm_bindgen]
pub struct SpaceTimeMachine {
    machine: Machine,
    space_time: SampledSpaceTime,
}

// impl iterator for spacetime machine
#[allow(clippy::missing_trait_methods)]
impl Iterator for SpaceTimeMachine {
    type Item = ();

    //cmki
    #[inline(never)]
    fn next(&mut self) -> Option<Self::Item> {
        self.machine.next()?;
        self.space_time.snapshot(&self.machine);
        Some(())
    }
}

#[wasm_bindgen]
#[allow(clippy::min_ident_chars)]
impl SpaceTimeMachine {
    #[wasm_bindgen(constructor)]
    pub fn from_str(
        s: &str,
        goal_x: u32,
        goal_y: u32,
        x_smoothness: u8,
        y_smoothness: u8,
        buffer1_count: u8,
    ) -> Result<Self, String> {
        Ok(Self {
            machine: Machine::from_string(s)?,
            space_time: SampledSpaceTime::new(
                goal_x,
                goal_y,
                PowerOfTwo::from_exp(x_smoothness),
                PowerOfTwo::from_exp(y_smoothness),
                PowerOfTwo::from_exp(buffer1_count),
            ),
        })
    }

    #[wasm_bindgen(js_name = "nth")]
    pub fn nth_js(&mut self, n: u64) -> bool {
        for _ in 0..=n {
            if self.next().is_none() {
                return false;
            }
        }
        true
    }

    #[wasm_bindgen(js_name = "step_for_secs")]
    #[allow(clippy::shadow_reuse)]
    pub fn step_for_secs_js(
        &mut self,
        seconds: f32,
        early_stop: Option<u64>,
        loops_per_time_check: u64,
    ) -> bool {
        let start = Instant::now();
        let step_count = self.step_count();

        // no early stop
        let Some(early_stop) = early_stop else {
            if step_count == 1 {
                for _ in 0..loops_per_time_check.saturating_sub(1) {
                    if self.next().is_none() {
                        return false;
                    }
                }
                if start.elapsed().as_secs_f32() >= seconds {
                    return true;
                }
            }
            loop {
                for _ in 0..loops_per_time_check {
                    if self.next().is_none() {
                        return false;
                    }
                }
                if start.elapsed().as_secs_f32() >= seconds {
                    return true;
                }
            }
        };

        // early stop
        if step_count >= early_stop {
            return false;
        }
        let mut remaining = early_stop - step_count;
        if step_count == 1 {
            let loops_per_time2 = loops_per_time_check.saturating_sub(1).min(remaining);
            for _ in 0..loops_per_time2 {
                if self.next().is_none() {
                    return false;
                }
            }
            if start.elapsed().as_secs_f32() >= seconds {
                return true;
            }
            remaining -= loops_per_time2;
        }
        while remaining > 0 {
            let loops_per_time2 = loops_per_time_check.min(remaining);
            for _ in 0..loops_per_time2 {
                if self.next().is_none() {
                    return false;
                }
            }
            if start.elapsed().as_secs_f32() >= seconds {
                return true;
            }
            remaining -= loops_per_time2;
        }
        true
    }

    #[wasm_bindgen]
    //cmki
    #[inline(never)]
    #[must_use]
    pub fn png_data(&mut self) -> Vec<u8> {
        self.space_time
            .to_png()
            .unwrap_or_else(|e| format!("{e:?}").into_bytes())
    }

    #[wasm_bindgen]
    //cmki
    #[inline(never)]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn step_count(&self) -> u64 {
        self.space_time.step_index + 1
    }

    #[wasm_bindgen]
    //cmki
    #[inline(never)]
    #[must_use]
    pub fn count_ones(&self) -> u32 {
        self.machine.count_ones()
    }

    #[wasm_bindgen]
    //cmki
    #[inline(never)]
    #[must_use]
    pub fn is_halted(&self) -> bool {
        self.machine.is_halted()
    }
}

/// A logarithmic iterator that generates `num_frames` steps between 0 and `max_value`, inclusive.
/// The steps are spaced approximately logarithmically, but constrained to integers.
pub struct LogStepIterator {
    current_frame: u32,
    total_frames: u32,
    max_value: u64,
}

impl LogStepIterator {
    //cmki
    #[inline(never)]
    #[must_use]
    pub const fn new(max_value: u64, total_frames: u32) -> Self {
        Self {
            current_frame: 0,
            total_frames,
            max_value,
        }
    }
}

#[allow(
    clippy::missing_trait_methods,
    clippy::min_ident_chars,
    clippy::float_cmp
)]
impl Iterator for LogStepIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_frame >= self.total_frames {
            return None;
        }

        // Normalized frame index from 0 to 1
        let t = self.current_frame as f64 / (self.total_frames - 1) as f64;

        // Apply logarithmic-like spacing using an exponential function
        let value = if t == 0.0 {
            0
        } else if t == 1.0 {
            self.max_value - 1
        } else {
            let log_value = ((self.max_value as f64).ln() * t).exp();
            log_value.round().min((self.max_value - 1) as f64) as u64
        };

        self.current_frame += 1;
        Some(value)
    }
}

//cmki
#[inline(never)]
fn fast_is_even<T>(x: T) -> bool
where
    T: Copy + core::ops::BitAnd<Output = T> + core::ops::Sub<Output = T> + From<u8> + PartialEq,
{
    (x & T::from(1)) == T::from(0)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PowerOfTwo(u8);

impl core::ops::Div for PowerOfTwo {
    type Output = Self;

    /// Will always be at least 1.
    //cmki
    #[inline(never)]
    fn div(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.0 >= rhs.0,
            "Divisor must be less than or equal to dividend"
        );
        self.saturating_div(rhs)
    }
}

impl core::ops::Mul<usize> for PowerOfTwo {
    type Output = usize;

    //cmki
    #[inline(never)]
    fn mul(self, rhs: usize) -> Self::Output {
        // Multiply rhs by 2^(self.0)
        // This is equivalent to shifting rhs left by self.0 bits.
        rhs * (1usize << self.0)
    }
}

// cmk make the auto constructor so private that it can't be used w/ modules, so that new check is run.

impl PowerOfTwo {
    /// The smallest valid `Smoothness` value, representing `2^0 = 1`.
    pub const ONE: Self = Self(0);
    pub const TWO: Self = Self(1);
    pub const FOUR: Self = Self(2);
    pub const EIGHT: Self = Self(3);
    pub const SIXTEEN: Self = Self(4);
    pub const THIRTY_TWO: Self = Self(5);
    pub const SIXTY_FOUR: Self = Self(6);
    pub const MIN: Self = Self(0);
    pub const MAX: Self = Self(63);

    //cmki
    #[inline(never)]
    #[must_use]
    pub fn offset_to_align(self, len: usize) -> usize {
        debug_assert!(
            (self.0 as u32) < usize::BITS,
            "Cannot shift left by self.0 = {} for usize::BITS = {}, which would overflow.",
            self.0,
            usize::BITS
        );
        len.wrapping_neg() & ((1 << self.0) - 1)
    }

    //cmki
    #[inline(never)]
    #[must_use]
    pub const fn from_exp(value: u8) -> Self {
        debug_assert!(value <= Self::MAX.0, "Value must be 63 or less");
        Self(value)
    }

    //cmki
    #[inline(never)]
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        1 << self.0
    }

    //cmki
    #[inline(never)]
    #[must_use]
    pub const fn saturating_div(self, rhs: Self) -> Self {
        // Subtract exponents; if the subtrahend is larger, saturate to 0 aks One
        Self(self.0.saturating_sub(rhs.0))
    }

    //cmki
    #[inline(never)]
    pub const fn assign_saturating_div_two(&mut self) {
        self.0 = self.0.saturating_sub(1);
    }

    //cmki
    #[inline(never)]
    #[must_use]
    pub const fn double(self) -> Self {
        debug_assert!(self.0 < Self::MAX.0, "Value must be 63 or less");
        Self(self.0 + 1)
    }

    //cmki
    #[inline(never)]
    #[must_use]
    pub fn as_usize(self) -> usize {
        let bits = core::mem::size_of::<usize>() * 8;
        debug_assert!(
            (self.0 as usize) < bits,
            "Exponent {} too large for usize ({} bits)",
            self.0,
            bits
        );
        1 << self.0
    }

    // from u64
    #[allow(clippy::missing_panics_doc)]
    //cmki
    #[inline(never)]
    #[must_use]
    pub fn from_u64(value: u64) -> Self {
        debug_assert!(value.is_power_of_two(), "Value must be a power of two");
        Self::from_exp(value.trailing_zeros() as u8)
    }

    //cmki
    #[inline(never)]
    #[must_use]
    pub const fn from_usize(value: usize) -> Self {
        debug_assert!(value.is_power_of_two(), "Value must be a power of two");
        Self::from_exp(value.trailing_zeros() as u8)
    }

    // //cmki
    // #[inline(never)]
    // #[must_use]
    // pub const fn from_usize_const(value: usize) -> Self {
    //     debug_assert!(value.is_power_of_two(), "Value must be a power of two");
    //     Self(value.trailing_zeros() as u8)
    // }

    //cmki
    #[inline(never)]
    #[must_use]
    pub const fn log2(self) -> u8 {
        self.0
    }

    //cmki
    #[inline(never)]
    pub fn rem_into_u64<T>(self, x: T) -> T
    where
        T: Copy
            + core::ops::BitAnd<Output = T>
            + core::ops::Sub<Output = T>
            + From<u64>
            + PartialEq,
    {
        x & (T::from(self.as_u64()) - T::from(1u64))
    }

    #[inline(never)]
    pub fn rem_into_usize<T>(self, x: T) -> T
    where
        T: Copy
            + core::ops::BitAnd<Output = T>
            + core::ops::Sub<Output = T>
            + From<usize>
            + PartialEq,
    {
        x & (T::from(self.as_usize()) - T::from(1usize))
    }

    //cmki
    #[inline(never)]
    #[must_use]
    pub fn rem_euclid_into(self, dividend: i64) -> i64 {
        let divisor = 1i64 << self.0; // Compute 2^n
        debug_assert!(divisor > 0, "divisor must be a power of two");
        let mask = divisor - 1;
        let remainder = dividend & mask;

        // If the remainder is negative, make it positive by adding divisor
        remainder + (divisor & (remainder >> Self::MAX.0))
    }

    //cmki
    #[inline(never)]
    #[must_use]
    pub fn div_ceil_into<T>(self, other: T) -> T
    where
        T: Copy
            + core::ops::Add<Output = T>
            + core::ops::Sub<Output = T>
            + core::ops::Shl<u8, Output = T>
            + core::ops::Shr<u8, Output = T>
            + From<u8>,
    {
        let one = T::from(1);
        let two_pow = one << self.0;
        (other + two_pow - one) >> self.0
    }

    //cmki
    #[inline(never)]
    pub fn divide_into<T>(self, x: T) -> T
    where
        T: Copy + core::ops::Shr<u8, Output = T>,
    {
        x >> self.0
    }

    //cmki
    #[inline(never)]
    #[must_use]
    pub const fn divides_u64(self, x: u64) -> bool {
        // If x is divisible by 2^(self.0), shifting right then left recovers x.
        (x >> self.0) << self.0 == x
    }

    //cmki
    #[inline(never)]
    #[must_use]
    pub const fn divides_i64(self, x: i64) -> bool {
        (x >> self.0) << self.0 == x
    }

    #[inline(never)]
    #[must_use]
    pub const fn divides_usize(self, x: usize) -> bool {
        (x >> self.0) << self.0 == x
    }

    // //cmki
    // #[inline(never)]
    // #[must_use]
    // pub const fn divides_smoothness(self, other: Self) -> bool {
    //     self.0 <= other.0
    // }
}

//cmki
#[inline(never)]
/// This returns the largest power of two that is less than or equal
/// to the input number x.
const fn prev_power_of_two(x: usize) -> usize {
    debug_assert!(x > 0, "x must be greater than 0");
    1usize << (usize::BITS as usize - x.leading_zeros() as usize - 1)
}

#[must_use]
pub fn average_with_iterators(values: &AVec<u8>, step: PowerOfTwo) -> AVec<u8> {
    let mut result = AVec::with_capacity(ALIGN, step.div_ceil_into(values.len()));

    // Process complete chunks
    let chunk_iter = values.chunks_exact(step.as_usize());
    let remainder = chunk_iter.remainder();

    for chunk in chunk_iter {
        let sum: u32 = chunk.iter().map(|&x| x as u32).sum();
        let average = step.divide_into(sum * 255) as u8;
        result.push(average);
    }

    // Handle the remainder - pad with zeros
    if !remainder.is_empty() {
        let sum: u32 = remainder.iter().map(|&x| x as u32).sum();
        // We need to divide by step size, not remainder.len()
        let average = step.divide_into(sum * 255) as u8;
        result.push(average);
    }

    result
}

#[allow(clippy::missing_panics_doc, clippy::integer_division_remainder_used)]
#[must_use]
// cmk0000000 if this is used, do full correctness check
pub fn average_with_simd_rayon<const LANES: usize>(values: &AVec<u8>, step: PowerOfTwo) -> AVec<u8>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    assert!(
        { LANES } <= step.as_usize() && { LANES } <= { ALIGN },
        "LANES must be less than or equal to step and alignment"
    );

    let values_len = values.len();
    let capacity = step.div_ceil_into(values_len);
    let mut result = AVec::with_capacity(ALIGN, capacity);
    result.resize(result.capacity(), 0); // Pre-fill with zeros
    let lanes = PowerOfTwo::from_exp(LANES.trailing_zeros() as u8);
    let rayon_threads = current_num_threads();
    let result_chunk_size = (capacity.div_ceil(rayon_threads * LANES)) * LANES;
    let input_chunk_size = result_chunk_size * step.as_usize();
    // Process SIMD chunks directly (each chunk is N elements)
    let lanes_per_chunk = step.saturating_div(lanes);

    result
        .par_chunks_mut(result_chunk_size)
        .zip(values.par_chunks(input_chunk_size))
        .for_each(|(result_chunk, input_chunk)| {
            let (prefix, chunks, _suffix) = input_chunk.as_simd::<LANES>();

            // Since we're using AVec with 64-byte alignment, the prefix should be empty
            debug_assert!(prefix.is_empty(), "Expected empty prefix due to alignment");

            if lanes_per_chunk == PowerOfTwo::ONE {
                for (average, chunk) in result_chunk.iter_mut().zip(chunks.iter()) {
                    let sum = chunk.reduce_sum() as u32;
                    *average = step.divide_into(sum * 255) as u8;
                }
            } else {
                let mut chunk_iter = chunks.chunks_exact(lanes_per_chunk.as_usize());

                // Process complete chunks
                for (average, sub_chunk) in result_chunk.iter_mut().zip(&mut chunk_iter) {
                    // Sum the values within the vector - values are just 0 or 1
                    let sum: u32 = sub_chunk
                        .iter()
                        .map(|chunk| chunk.reduce_sum() as u32)
                        .sum();
                    *average = step.divide_into(sum * 255) as u8;
                }
            }

            // How many elements are unprocessed?
            let unused_items = step.rem_into_usize(values_len);
            if unused_items > 0 {
                // sum the last missing_items
                let sum: u32 = values
                    .iter()
                    .rev()
                    .take(unused_items)
                    .map(|&x| x as u32)
                    .sum();
                *(result_chunk.last_mut().unwrap()) = step.divide_into(sum * 255) as u8;
            }
        });

    result
}

#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn average_with_simd<const LANES: usize>(values: &AVec<u8>, step: PowerOfTwo) -> AVec<u8>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    assert!(
        { LANES } <= step.as_usize() && { LANES } <= { ALIGN },
        "LANES must be less than or equal to step and alignment"
    );

    let values_len = values.len();
    let mut result = AVec::with_capacity(ALIGN, step.div_ceil_into(values_len));
    let lanes = PowerOfTwo::from_exp(LANES.trailing_zeros() as u8);

    let (prefix, chunks, _suffix) = values.as_slice().as_simd::<LANES>();

    // Since we're using AVec with 64-byte alignment, the prefix should be empty
    debug_assert!(prefix.is_empty(), "Expected empty prefix due to alignment");
    // Process SIMD chunks directly (each chunk is N elements)
    let lanes_per_chunk = step.saturating_div(lanes);

    if lanes_per_chunk == PowerOfTwo::ONE {
        for chunk in chunks {
            let sum = chunk.reduce_sum() as u32;
            let average = step.divide_into(sum * 255) as u8;
            result.push(average);
        }
    } else {
        let mut chunk_iter = chunks.chunks_exact(lanes_per_chunk.as_usize());

        // Process complete chunks
        for sub_chunk in &mut chunk_iter {
            // Sum the values within the vector - values are just 0 or 1
            let sum: u32 = sub_chunk
                .iter()
                .map(|chunk| chunk.reduce_sum() as u32)
                .sum();
            let average = step.divide_into(sum * 255) as u8;
            result.push(average);
        }
    }

    // How many elements are unprocessed?
    let unused_items = step.rem_into_usize(values_len);
    if unused_items > 0 {
        // sum the last missing_items
        let sum: u32 = values
            .iter()
            .rev()
            .take(unused_items)
            .map(|&x| x as u32)
            .sum();
        let average = step.divide_into(sum * 255) as u8;
        result.push(average);
    }

    result
}

#[cfg(test)]
mod tests {
    use std::fs;

    use wasm_bindgen_test::wasm_bindgen_test;
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
    use super::*;

    #[test]
    fn bb5_champ() -> Result<(), Error> {
        let mut machine: Machine = BB5_CHAMP.parse()?;

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

    #[wasm_bindgen_test]
    #[test]
    fn bb5_champ_js() -> Result<(), String> {
        let mut machine: Machine = Machine::from_string(BB5_CHAMP)?;

        let early_stop_some = true;
        let early_stop_number = 50_000_000;
        let step_count = machine.count_js(early_stop_some, early_stop_number);

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

    /// See <https://en.wikipedia.org/wiki/Busy_beaver>
    #[allow(clippy::shadow_reuse, clippy::integer_division_remainder_used)]
    #[test]
    fn bb5_champ_space_time_native() -> Result<(), Error> {
        let mut machine: Machine = BB5_CHAMP.parse()?; // cmk
        // let mut machine: Machine = BB6_CONTENDER.parse()?;

        let goal_x: u32 = 1000;
        let goal_y: u32 = 1000;
        let x_smoothness: PowerOfTwo = PowerOfTwo::ONE;
        let y_smoothness: PowerOfTwo = PowerOfTwo::ONE;
        let buffer1_count: PowerOfTwo = PowerOfTwo::ONE; // cmk000000
        let mut sample_space_time =
            SampledSpaceTime::new(goal_x, goal_y, x_smoothness, y_smoothness, buffer1_count);

        let early_stop = Some(10_500_000);
        // let early_stop = Some(1_000_000);
        let debug_interval = Some(1_000_000);

        while machine.next().is_some()
            && early_stop.is_none_or(|early_stop| sample_space_time.step_index + 1 < early_stop)
        {
            if debug_interval
                .is_none_or(|debug_interval| sample_space_time.step_index % debug_interval == 0)
            {
                println!(
                    "Step {}: {:?},\t{}",
                    sample_space_time.step_index.separate_with_commas(),
                    machine,
                    machine.tape.index_range_to_string(-10..=10)
                );
            }

            sample_space_time.snapshot(&machine);
            // let _ = sample_space_time.to_png();
        }

        let png_data = sample_space_time.to_png()?;
        fs::write("tests/expected/test.png", &png_data).unwrap(); // cmk handle error

        println!(
            "Final: Steps {}: {:?}, #1's {}",
            sample_space_time.step_index.separate_with_commas(),
            machine,
            machine.tape.count_ones()
        );

        if early_stop.is_none() {
            assert_eq!(sample_space_time.step_index, 47_176_870);
            assert_eq!(machine.tape.count_ones(), 4098);
            assert_eq!(machine.state, 7);
            assert_eq!(machine.tape_index, -12242);
        }

        Ok(())
    }

    #[wasm_bindgen_test]
    #[test]
    fn bb5_champ_space_time_js() -> Result<(), String> {
        let program_string = BB5_CHAMP;
        let goal_x: u32 = 1000;
        let goal_y: u32 = 1000;
        let x_smoothness: PowerOfTwo = PowerOfTwo::ONE;
        let y_smoothness: PowerOfTwo = PowerOfTwo::ONE;
        let buffer1_count: PowerOfTwo = PowerOfTwo::ONE;
        let n = 1_000_000;
        let mut space_time_machine = SpaceTimeMachine::from_str(
            program_string,
            goal_x,
            goal_y,
            x_smoothness.log2(),
            y_smoothness.log2(),
            buffer1_count.log2(),
        )?;

        while space_time_machine.nth_js(n - 1) {
            println!(
                "Index {}: {:?}, #1's {}",
                space_time_machine
                    .space_time
                    .step_index
                    .separate_with_commas(),
                space_time_machine.machine,
                space_time_machine.machine.count_ones()
            );
        }

        println!(
            "Final: Steps {}: {:?}, #1's {}",
            space_time_machine
                .space_time
                .step_index
                .separate_with_commas(),
            space_time_machine.machine,
            space_time_machine.machine.count_ones()
        );

        let png_data = space_time_machine.png_data();
        fs::write("tests/expected/test_js.png", &png_data).map_err(|error| error.to_string())?;

        assert_eq!(space_time_machine.space_time.step_index + 1, 47_176_870);
        assert_eq!(space_time_machine.machine.count_ones(), 4098);
        assert_eq!(space_time_machine.machine.state, 7);
        assert_eq!(space_time_machine.machine.tape_index, -12242);

        Ok(())
    }

    #[wasm_bindgen_test]
    #[test]
    fn seconds_bb5_champ_space_time_js() -> Result<(), String> {
        let program_string = BB5_CHAMP;
        let goal_x: u32 = 1000;
        let goal_y: u32 = 1000;
        let x_smoothness: PowerOfTwo = PowerOfTwo::ONE;
        let y_smoothness: PowerOfTwo = PowerOfTwo::ONE;
        let buffer1_count: PowerOfTwo = PowerOfTwo::ONE; // cmk000000
        let seconds = 0.25;
        let mut space_time_machine = SpaceTimeMachine::from_str(
            program_string,
            goal_x,
            goal_y,
            x_smoothness.log2(),
            y_smoothness.log2(),
            buffer1_count.log2(),
        )?;

        while space_time_machine.step_for_secs_js(seconds, None, 100_000) {
            println!(
                "Index {}: {:?}, #1's {}",
                space_time_machine
                    .space_time
                    .step_index
                    .separate_with_commas(),
                space_time_machine.machine,
                space_time_machine.machine.count_ones()
            );
        }

        println!(
            "Final: Steps {}: {:?}, #1's {}",
            space_time_machine
                .space_time
                .step_index
                .separate_with_commas(),
            space_time_machine.machine,
            space_time_machine.machine.count_ones()
        );

        let png_data = space_time_machine.png_data();
        fs::write("tests/expected/test2_js.png", &png_data)
            .map_err(|error: std::io::Error| error.to_string())?;

        assert_eq!(space_time_machine.space_time.step_index + 1, 47_176_870);
        assert_eq!(space_time_machine.machine.count_ones(), 4098);
        assert_eq!(space_time_machine.machine.state, 7);
        assert_eq!(space_time_machine.machine.tape_index, -12242);

        Ok(())
    }

    #[wasm_bindgen_test]
    #[test]
    fn machine_7_135_505() -> Result<(), Error> {
        let _machine_a: Machine = MACHINE_7_135_505_A.parse()?;
        let _machine_b: Machine = MACHINE_7_135_505_B.parse()?;
        Ok(())
    }

    // Create a test that runs bb5 champ to halting and then prints the time it took
    // to run the test
    // cmk which of these should be bindgen tests?
    #[wasm_bindgen_test]
    #[test]
    fn bb5_champ_time() {
        let start = std::time::Instant::now();
        let step_count = 1 + BB5_CHAMP.parse::<Machine>().unwrap().count();
        let duration = start.elapsed();
        println!(
            "Steps: {}, Duration: {:?}",
            step_count.separate_with_commas(),
            duration
        );
        assert_eq!(step_count, 47_176_870);
    }

    #[test]
    fn benchmark1() -> Result<(), String> {
        let start = std::time::Instant::now();
        let program_string = BB6_CONTENDER;
        let goal_x: u32 = 360;
        let goal_y: u32 = 432;
        let x_smoothness: PowerOfTwo = PowerOfTwo::ONE;
        let y_smoothness: PowerOfTwo = PowerOfTwo::ONE;
        let buffer1_count: PowerOfTwo = PowerOfTwo::ONE; // cmk0000000
        let n = 500_000_000;
        let mut space_time_machine = SpaceTimeMachine::from_str(
            program_string,
            goal_x,
            goal_y,
            x_smoothness.log2(),
            y_smoothness.log2(),
            buffer1_count.log2(),
        )?;

        space_time_machine.nth_js(n - 1);

        println!("Elapsed: {:?}", start.elapsed());

        println!(
            "Final: Steps {}: {:?}, #1's {}",
            space_time_machine
                .space_time
                .step_index
                .separate_with_commas(),
            space_time_machine.machine,
            space_time_machine.machine.count_ones()
        );

        assert_eq!(space_time_machine.space_time.step_index, n);
        assert_eq!(space_time_machine.machine.count_ones(), 10669);
        assert_eq!(space_time_machine.machine.state, 1);
        assert_eq!(space_time_machine.machine.tape_index, 34054);

        // cmk LATER what is one method png_data and another to to_png?
        let start2 = std::time::Instant::now();
        let png_data = space_time_machine.png_data();
        fs::write("tests/expected/bench.png", &png_data).unwrap(); // cmk handle error
        println!("Elapsed png: {:?}", start2.elapsed());
        Ok(())
    }

    #[allow(clippy::shadow_reuse)]
    #[test]
    #[wasm_bindgen_test]
    fn benchmark2() -> Result<(), String> {
        // let start = std::time::Instant::now();
        let early_stop = Some(1_000_000_000);

        let program_string = BB6_CONTENDER;
        let goal_x: u32 = 360;
        let goal_y: u32 = 432;
        let x_smoothness: PowerOfTwo = PowerOfTwo::from_exp(0);
        let y_smoothness: PowerOfTwo = PowerOfTwo::from_exp(0);
        let buffer1_count: PowerOfTwo = PowerOfTwo::ONE; // cmk0000000
        let mut space_time_machine = SpaceTimeMachine::from_str(
            program_string,
            goal_x,
            goal_y,
            x_smoothness.log2(),
            y_smoothness.log2(),
            buffer1_count.log2(),
        )?;

        let chunk_size = 100_000_000;
        let mut total_steps = 1; // Start at 1 since first step is already taken

        loop {
            if early_stop.is_some_and(|early_stop| total_steps >= early_stop) {
                break;
            }

            // Calculate next chunk size
            let next_chunk = if total_steps == 1 {
                chunk_size - 1
            } else {
                chunk_size
            };

            let next_chunk = early_stop.map_or(next_chunk, |early_stop| {
                let remaining = early_stop - total_steps;
                remaining.min(next_chunk)
            });

            // Run the next chunk
            let continues = space_time_machine.nth_js(next_chunk - 1);
            total_steps += next_chunk;

            // Send intermediate update
            println!(
                "intermediate: {:?} Steps {}: {:?}, #1's {}",
                0,
                // start.elapsed(),
                space_time_machine
                    .space_time
                    .step_index
                    .separate_with_commas(),
                space_time_machine.machine,
                space_time_machine.machine.count_ones()
            );

            // let _png_data = space_time_machine.png_data();

            // Exit if machine halted
            if !continues {
                break;
            }
        }

        // Send final result

        println!(
            "Final: {:?} Steps {}: {:?}, #1's {}",
            0, // start.elapsed(),
            space_time_machine
                .space_time
                .step_index
                .separate_with_commas(),
            space_time_machine.machine,
            space_time_machine.machine.count_ones(),
        );

        // // cmk LATER what is one method png_data and another to to_png?
        // let start = std::time::Instant::now();
        // let png_data = space_time_machine.png_data();
        // fs::write("tests/expected/bench2.png", &png_data).unwrap(); // cmk handle error
        // println!("Elapsed png: {:?}", start.elapsed());
        Ok(())
    }

    // #[test]
    fn benchmark3() -> Result<(), String> {
        println!("Smoothness\tSteps\tOnes\tTime(ms)");

        for smoothness in 0..=63 {
            let start = std::time::Instant::now();
            let program_string = BB5_CHAMP;
            let goal_x: u32 = 360;
            let goal_y: u32 = 432;
            let x_smoothness = PowerOfTwo::from_exp(smoothness);
            let y_smoothness = PowerOfTwo::from_exp(smoothness);
            let buffer1_count = PowerOfTwo::ONE; // cmk0000000

            let mut space_time_machine = SpaceTimeMachine::from_str(
                program_string,
                goal_x,
                goal_y,
                x_smoothness.log2(),
                y_smoothness.log2(),
                buffer1_count.log2(),
            )?;

            // Run to completion
            while space_time_machine.nth_js(1_000_000 - 1) {}

            let elapsed = start.elapsed().as_millis();
            println!(
                "{}\t{}\t{}\t{}",
                smoothness,
                space_time_machine.step_count(),
                space_time_machine.count_ones(),
                elapsed
            );

            // Generate PNG for first and last iteration
            if smoothness == 0 || smoothness == 63 {
                let png_data = space_time_machine.png_data();
                fs::write(
                    format!("tests/expected/bench3_smooth{smoothness}.png"),
                    &png_data,
                )
                .map_err(|error| error.to_string())?;
            }
        }

        Ok(())
    }

    #[allow(clippy::shadow_reuse)]
    #[test]
    #[wasm_bindgen_test]
    fn benchmark63() -> Result<(), String> {
        // let start = std::time::Instant::now();

        // let early_stop = Some(10_000_000_000);
        // let chunk_size = 10_000_000;
        let early_stop = Some(50_000_000);
        let chunk_size = 5_000_000;
        // let early_stop = Some(5_000_000);
        // let chunk_size = 500_000;

        let program_string = BB6_CONTENDER;
        let goal_x: u32 = 360;
        let goal_y: u32 = 432;
        let x_smoothness: PowerOfTwo = PowerOfTwo::from_exp(63); // cmk0000 63);
        let y_smoothness: PowerOfTwo = PowerOfTwo::from_exp(63);
        let buffer1_count: PowerOfTwo = PowerOfTwo::from_exp(0);
        let mut space_time_machine = SpaceTimeMachine::from_str(
            program_string,
            goal_x,
            goal_y,
            x_smoothness.log2(),
            y_smoothness.log2(),
            buffer1_count.log2(),
        )?;

        let mut total_steps = 1; // Start at 1 since first step is already taken

        loop {
            if early_stop.is_some_and(|early_stop| total_steps >= early_stop) {
                break;
            }

            // Calculate next chunk size
            let next_chunk = if total_steps == 1 {
                chunk_size - 1
            } else {
                chunk_size
            };

            let next_chunk = early_stop.map_or(next_chunk, |early_stop| {
                let remaining = early_stop - total_steps;
                remaining.min(next_chunk)
            });

            // Run the next chunk
            let continues = space_time_machine.nth_js(next_chunk - 1);
            total_steps += next_chunk;

            // Send intermediate update
            println!(
                "intermediate: {:?} Steps {}: {:?}, #1's {}",
                0,
                // start.elapsed(),
                space_time_machine
                    .space_time
                    .step_index
                    .separate_with_commas(),
                space_time_machine.machine,
                space_time_machine.machine.count_ones()
            );

            // let _png_data = space_time_machine.png_data();

            // Exit if machine halted
            if !continues {
                break;
            }
        }

        // Send final result

        println!(
            "Final: {:?} Steps {}: {:?}, #1's {}",
            0, // start.elapsed(),
            space_time_machine
                .space_time
                .step_index
                .separate_with_commas(),
            space_time_machine.machine,
            space_time_machine.machine.count_ones(),
        );

        // cmk LATER what is one method png_data and another to to_png?
        let start = std::time::Instant::now();
        let png_data = space_time_machine.png_data();
        fs::write("tests/expected/bench63.png", &png_data).unwrap(); // cmk handle error
        println!("Elapsed png: {:?}", start.elapsed());
        Ok(())
    }

    #[allow(clippy::shadow_unrelated)]
    #[test]
    fn test_average() {
        let values = AVec::from_slice(ALIGN, &[0, 0, 0, 1, 1, 0, 1, 1, 1]);

        let step = PowerOfTwo::ONE;
        let expected = &[0, 0, 0, 255, 255, 0, 255, 255, 255];
        let result = average_with_iterators(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<1>(&values, step);
        assert_eq!(result.as_slice(), expected);

        let step = PowerOfTwo::TWO;
        let expected = &[0, 127, 127, 255, 127];
        let result = average_with_iterators(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<1>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<2>(&values, step);
        assert_eq!(result.as_slice(), expected);
        // Expected to panic
        // let result = average_with_simd::<4>(&values, step);
        // assert_eq!(result.as_slice(), expected);

        let step = PowerOfTwo::FOUR;
        let expected = &[63, 191, 63];
        let result = average_with_iterators(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<1>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<2>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<4>(&values, step);
        assert_eq!(result.as_slice(), expected);

        let step = PowerOfTwo::EIGHT;
        let expected = &[127, 31];
        let result = average_with_iterators(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<1>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<2>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<4>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<8>(&values, step);
        assert_eq!(result.as_slice(), expected);

        let step = PowerOfTwo::SIXTEEN;
        let expected = &[79];
        let result = average_with_iterators(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<1>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<2>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<4>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<8>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<16>(&values, step);
        assert_eq!(result.as_slice(), expected);

        let step = PowerOfTwo::THIRTY_TWO;
        let expected = &[39];
        let result = average_with_iterators(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<1>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<2>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<4>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<8>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<16>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<32>(&values, step);
        assert_eq!(result.as_slice(), expected);

        let step = PowerOfTwo::SIXTY_FOUR;
        let expected = &[19];
        let result = average_with_iterators(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<1>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<2>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<4>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<8>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<16>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<32>(&values, step);
        assert_eq!(result.as_slice(), expected);
        let result = average_with_simd::<64>(&values, step);
        assert_eq!(result.as_slice(), expected);

        // Rayon is slower, but is it correct?
        let result = average_with_simd_rayon::<64>(&values, step);
        assert_eq!(result.as_slice(), expected);
    }
}
