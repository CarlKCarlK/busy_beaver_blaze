use core::{fmt, panic};
use derive_more::derive::Display;
use derive_more::Error as DeriveError;
use itertools::Itertools;
use num_traits::PrimInt;
use png::{BitDepth, ColorType, Encoder};
use std::ops::Shr;
#[cfg(not(target_arch = "wasm32"))]
use std::str::FromStr;
use thousands::Separable;
use wasm_bindgen::prelude::*;

// cmk000 is the image size is a power of 2, then don't filter ???

const BB2_CHAMP: &str = "
	A	B
0	1RB	1LA
1	1LB	1RH
";

const BB3_CHAMP: &str = "
	A	B	C
0	1RB	0RC	1LC
1	1RH	1RB	1LA
";
const BB4_CHAMP: &str = "
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

const Machine_7_135_505_A: &str = "   
0	1
A	1RB	0LD
B	1RC	---
C	1LD	1RA
D	1RE	1LC
E	0LA	0RE
";
const Machine_7_135_505_B: &str = "1RB0LD_1RC---_1LD1RA_1RE1LC_0LA0RE";

#[derive(Default, Debug)]
struct Tape {
    nonnegative: Vec<u8>,
    negative: Vec<u8>,
}

impl Tape {
    #[inline]
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

    #[inline]
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
            .map(|&x| (x == 1) as usize)
            .sum()
    }

    fn index_range_to_string(&self, range: std::ops::RangeInclusive<i64>) -> String {
        let mut s = String::new();
        for i in range {
            s.push_str(&self.read(i).to_string());
        }
        s
    }

    #[inline]
    pub fn min_index(&self) -> i64 {
        -(self.negative.len() as i64)
    }

    #[inline]
    pub fn max_index(&self) -> i64 {
        self.nonnegative.len() as i64 - 1
    }

    #[inline]
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
    pub fn from_string(input: &str) -> Result<Machine, String> {
        input.parse().map_err(|e| format!("{:?}", e))
    }

    #[wasm_bindgen]
    pub fn step(&mut self) -> bool {
        self.next().is_some()
    }

    #[wasm_bindgen]
    pub fn count_ones(&self) -> u32 {
        self.tape.count_ones() as u32
    }

    #[wasm_bindgen]
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

        Ok(Machine {
            tape: Tape::default(),
            tape_index: 0,
            program,
            state: 0,
        })
    }
}

impl fmt::Debug for Machine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Machine {{ state: {}, tape_index: {}}}",
            self.state, self.tape_index
        )
    }
}

impl Iterator for Machine {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        let program = &self.program;
        let input = self.tape.read(self.tape_index);
        let per_input = &program.state_to_symbol_to_action[self.state as usize][input as usize];
        self.tape.write(self.tape_index, per_input.next_symbol);
        self.tape_index += per_input.direction as i64;
        self.state = per_input.next_state;
        if self.state < program.state_count {
            Some(())
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct Program {
    state_count: u8,
    symbol_count: u8,
    state_to_symbol_to_action: Vec<Vec<Action>>,
}

// impl a parser for the strings like this
// "   A	B	C	D	E
// 0	1RB	1RC	1RD	1LA	1RH
// 1	1LC	1RB	0LE	1LD	0LA"
impl FromStr for Program {
    type Err = Error;

    #[allow(clippy::assertions_on_constants)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let count_lines = s.lines().count();
        let is_first_non_space_a_numeral = s
            .trim()
            .chars()
            .next()
            .map_or(false, |c| c.is_ascii_digit());

        match (count_lines, is_first_non_space_a_numeral) {
            (1, _) => Self::parse_standard_format(s),
            (2.., false) => Self::parse_symbol_to_state(s),
            (2.., true) => Self::parse_state_to_symbol(s),
            _ => Err(Error::UnexpectedFormat),
        }
    }
}

impl Program {
    fn parse_state(input: impl AsRef<str>) -> Result<char, Error> {
        // println!("cmk {:?}", input.as_ref());
        let mut chars = input.as_ref().chars();
        match (chars.next(), chars.next()) {
            (Some(c @ 'A'..='Z'), None) => Ok(c), // Ensure single uppercase letter
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

    #[allow(clippy::assertions_on_constants)]
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
                    .and_then(Program::parse_state)?;

                if state != state_again {
                    return Err(Error::UnexpectedState);
                }

                parts
                    .map(Program::parse_action)
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

        let symbol_count = state_to_symbol_to_action[0].len() as u8;
        if symbol_count == 0 {
            return Err(Error::InvalidSymbolsCount {
                expected: 1,
                got: 0,
            });
        }

        Ok(Program {
            state_count,
            symbol_count,
            state_to_symbol_to_action,
        })
    }

    #[allow(clippy::assertions_on_constants)]
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
                    .map(Program::parse_action)
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

        let symbol_count = state_to_symbol_to_action[0].len() as u8;
        if symbol_count == 0 {
            return Err(Error::InvalidSymbolsCount {
                expected: 1,
                got: 0,
            });
        }

        Ok(Program {
            state_count,
            symbol_count,
            state_to_symbol_to_action,
        })
    }

    #[allow(clippy::assertions_on_constants)]
    fn parse_symbol_to_state(s: &str) -> Result<Self, Error> {
        let mut lines = s.lines();

        // Skip empty lines at the beginning
        for line in lines.by_ref() {
            if !line.trim().is_empty() {
                break;
            }
        }

        // Create a vector of vectors, e.g. 2 x 5
        let mut vec_of_vec: Vec<Vec<Action>> = lines
            .enumerate()
            .map(|(symbol, line)| {
                let mut parts = line.split_whitespace();

                let symbol_again = parts.next().ok_or(Error::MissingField)?.parse::<u8>()?;
                if symbol != symbol_again as usize {
                    return Err(Error::UnexpectedSymbol);
                }

                parts
                    .map(Program::parse_action)
                    .collect::<Result<Vec<_>, _>>() // Collect and propagate any errors
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

        let state_count = vec_of_vec[0].len();
        if state_count == 0 {
            return Err(Error::InvalidStatesCount {
                expected: 1,
                got: 0,
            });
        }

        // Preallocate transposed vec_of_vec (state_count x symbol_count)
        let mut state_to_symbol_to_action: Vec<Vec<Action>> = (0..state_count)
            .map(|_| Vec::with_capacity(symbol_count))
            .collect();

        // Drain and fill the transposed matrix
        for mut row in vec_of_vec.drain(..) {
            if row.len() != state_count {
                return Err(Error::InvalidStatesCount {
                    expected: state_count,
                    got: row.len(),
                });
            }

            for (i, item) in row.drain(..).enumerate() {
                state_to_symbol_to_action[i].push(item); // Move item into transposed[i]
            }
        }

        Ok(Program {
            state_count: state_count as u8,
            symbol_count: symbol_count as u8,
            state_to_symbol_to_action,
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
    #[inline]
    fn debug_count(&mut self, debug_interval: usize) -> usize
    where
        Self: Sized + std::fmt::Debug, // ✅ Ensure Debug is implemented
    {
        let mut step_index = 0;

        println!("Step {}: {:?}", step_index.separate_with_commas(), self);

        while self.next().is_some() {
            step_index += 1;
            if step_index % debug_interval == 0 {
                println!("Step {}: {:?}", step_index.separate_with_commas(), self);
            }
        }

        step_index + 1 // turn last index into count
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

    #[display("Unexpected state encountered")]
    UnexpectedState,

    #[display("Invalid encoding encountered")]
    EncodingError,

    #[display("Unexpected format")]
    UnexpectedFormat,
}

// Implement conversions manually where needed
impl From<std::num::ParseIntError> for Error {
    fn from(err: std::num::ParseIntError) -> Self {
        Error::ParseIntError(err)
    }
}

#[derive(Debug, Default, Display, Copy, Clone)]
struct Pixel(u8);

impl Pixel {
    #[inline]
    fn merge_with_white(&mut self) {
        // inplace divide u8 by 2
        self.0 >>= 1;
    }

    #[inline]
    fn merge(&mut self, other: Self) {
        self.0 = (self.0 >> 1) + (other.0 >> 1) + ((self.0 & other.0) & 1);
    }

    // cmk00speed
    fn merge_slice_down_sample(slice: &[Self], empty_count: u64, down_step: Smoothness) -> Self {
        debug_assert!(down_step.divides_u64(slice.len() as u64), "real assert d1");
        let mut sum: u32 = 0;
        for i in (0..slice.len()).step_by(down_step.as_u64() as usize) {
            sum += slice[i].0 as u32;
        }
        let total_len = Smoothness::from_u64(slice.len() as u64 + empty_count);
        let count = total_len / down_step;
        let mean = count.divide_into(sum) as u8;
        Pixel(mean)
    }

    fn merge_slice_all(slice: &[Self], empty_count: i64) -> Self {
        let sum: u32 = slice.iter().map(|p| p.0 as u32).sum();
        let count = slice.len() + empty_count as usize;
        debug_assert!(count.is_power_of_two(), "Count must be a power of two");
        Pixel(Smoothness::from_u64(count as u64).divide_into(sum) as u8)
    }
}

impl From<u8> for Pixel {
    #[inline]
    fn from(value: u8) -> Self {
        debug_assert!(value <= 1, "Input value must be 0 or 1, got {}", value);
        Pixel(value * 255)
    }
}

const PIXEL_WHITE: Pixel = Pixel(0);

#[derive(Clone, Debug)]
struct Spaceline {
    sample: Smoothness,
    start: i64,
    pixels: Vec<Pixel>,
    time: u64,
    smoothness: Smoothness,
}

impl Spaceline {
    // cmk good name??
    fn new0(smoothness: Smoothness) -> Self {
        Spaceline {
            sample: Smoothness::ONE,
            start: 0,
            pixels: vec![PIXEL_WHITE; 1],
            time: 0,
            smoothness,
        }
    }

    fn resample_if_needed(&mut self, sample: Smoothness) {
        // Sampling & Averaging 2 --
        // When we merge rows, we sometimes need to squeeze the earlier row to
        // have the same sample rate as the later row.

        assert!(!self.pixels.is_empty(), "real assert a");
        assert!(
            self.sample.divides_i64(self.start),
            "Start must be a multiple of the sample rate",
        );
        if sample == self.sample {
            return;
        }
        let cells_to_add = fast_rem_euclid(self.start, sample.as_u64());
        let new_start = self.start - cells_to_add;
        let old_items_to_add = self.sample.divide_into(cells_to_add);
        let old_items_per_new = self.sample.divide_into(sample.as_u64()) as i64;
        assert!(
            sample >= self.sample && (old_items_per_new as u64).is_power_of_two(),
            "real assert 12"
        );
        let old_items_to_use = old_items_per_new - old_items_to_add;
        // println!(
        //     "cmk old_items_to_use {}, pixel.len {}",
        //     old_items_to_use,
        //     self.pixels.len()
        // );
        assert!(
            old_items_to_use <= self.pixels.len() as i64,
            "real assert d10"
        );

        let (_down_sample, down_step) = compute_sample_step(sample, self.smoothness);
        let pixel0 = Pixel::merge_slice_down_sample(
            &self.pixels[..old_items_to_use as usize],
            old_items_to_add as u64,
            down_step,
        );

        let mut new_index = 0usize;
        self.pixels[new_index] = pixel0;
        new_index += 1;
        let value_len = self.pixels.len() as i64;

        // cmk00speed
        for old_index in (old_items_to_use..value_len).step_by(old_items_per_new as usize) {
            let old_end = (old_index + old_items_to_use).min(value_len);
            let slice = &self.pixels[old_index as usize..old_end as usize];
            let old_items_to_add = (old_items_per_new - (old_end - old_index)) as u64;
            self.pixels[new_index] =
                Pixel::merge_slice_down_sample(slice, old_items_to_add, down_step);
            new_index += 1;
        }

        // trim the vector to the new length
        self.pixels.truncate(new_index);
        self.start = new_start;
        self.sample = sample;
    }

    fn merge(&mut self, other: Spaceline) {
        assert!(
            other.sample.divides_smoothness(other.sample),
            "real assert 6d"
        );
        assert!(self.time < other.time, "real assert 2");
        assert!(self.sample <= other.sample, "real assert 3");
        assert!(self.start >= other.start, "real assert 4");
        assert!(self.start >= other.start, "real assert 6b");
        self.resample_if_needed(other.sample);
        assert!(self.start >= other.start, "real assert 6c");

        let sample = other.sample;
        let mut values = other.pixels;
        let start = other.start;

        assert!(self.sample == sample, "real assert 5");
        assert!(self.start >= start, "real assert 6");
        assert!(sample.divides_i64(start - self.start), "real assert 7");
        let self_end = self.start + self.pixels.len() as i64 * sample.as_u64() as i64;
        let end = start + values.len() as i64 * sample.as_u64() as i64;
        assert!(sample.divides_i64(self_end - end), "real assert 8");
        assert!(self_end <= end, "real assert 9");

        let mut index = 0;
        // everything before self.start should get merged with merged with white
        // cmk00speed
        for _ in (start..self.start).step_by(sample.as_u64() as usize) {
            values[index].merge_with_white();
            index += 1;
        }

        // merge the overlapping part
        // cmk00speed
        for self_value in self.pixels.iter() {
            values[index].merge(*self_value);
            index += 1;
        }

        // merge the rest with white
        // cmk00speed
        let index2 = index;
        for _ in index2..values.len() {
            values[index].merge_with_white();
            index += 1;
        }

        self.pixels = values;
        self.start = start;
        self.sample = sample;
    }

    fn new(tape: &Tape, x_goal: u32, step_index: u64, x_smoothness: Smoothness) -> Self {
        // Sampling & Averaging 4 --
        let tape_width = tape.width();
        let tape_min_index = tape.min_index();
        let tape_max_index = tape.max_index();
        let x_sample = sample_rate(tape_width, x_goal);

        let sample_start: i64 = tape_min_index - fast_rem_euclid(tape_min_index, x_sample.as_u64());
        assert!(
            sample_start <= tape_min_index
                && x_sample.divides_i64(sample_start)
                && tape_min_index - sample_start < x_sample.as_u64() as i64,
            "real assert b1"
        );

        let mut pixels = Vec::with_capacity(x_goal as usize * 2);
        if x_sample == Smoothness::ONE {
            // For x_sample == 1, each step is just one pixel.
            for sample_index in sample_start..=tape_max_index {
                // Directly read and convert the pixel from the tape.
                pixels.push(tape.read(sample_index).into());
            }
        } else {
            // For x_sample > 1, process as before.

            // cmk00 for now, we allocate on the heap. May want to special case down_sample==1 or use SmallVec for 0 to say 8.

            let (down_sample, down_step) = compute_sample_step(x_sample, x_smoothness);
            let mut pixel_range = vec![Pixel(0); down_sample.as_u64() as usize];
            for sample_index in (sample_start..=tape_max_index).step_by(x_sample.as_u64() as usize)
            {
                // Create a temporary vector to hold x_sample pixels.
                for (i, pixel) in pixel_range.iter_mut().enumerate() {
                    *pixel = tape
                        .read(sample_index + (i as u64 * down_step.as_u64()) as i64)
                        .into();
                }
                pixels.push(Pixel::merge_slice_all(&pixel_range, 0));
            }
        }

        Spaceline {
            sample: x_sample,
            start: sample_start,
            pixels,
            time: step_index,
            smoothness: x_smoothness,
        }
    }
}

pub const MAX_POWER_OF_TWO_U64: u64 = 1 << 63; // cmk good name?

struct Spacelines {
    main: Vec<Spaceline>,
    buffer: Vec<Spaceline>,
}

impl Spacelines {
    fn new(smoothness: Smoothness) -> Self {
        Spacelines {
            main: vec![Spaceline::new0(smoothness)],
            buffer: Vec::new(),
        }
    }
    fn len(&self) -> usize {
        self.main.len() + if self.buffer.is_empty() { 0 } else { 1 }
    }

    fn get<'a>(&'a self, index: usize, last: &'a Spaceline) -> &'a Spaceline {
        if index == self.len() - 1 {
            last
        } else {
            &self.main[index]
        }
    }

    fn flush_buffer(&mut self) {
        // We now have a buffer that needs to be flushed at the end
        if !self.buffer.is_empty() {
            assert!(self.buffer.len() == 1, "real assert 13");
            self.main.push(self.buffer.pop().unwrap());
        }
    }

    #[inline]
    fn compress_average(&mut self) {
        assert!(self.buffer.is_empty(), "real assert b2");
        assert!(fast_is_even(self.main.len()), "real assert 11");
        // println!("cmk compress_average");

        self.main = self
            .main
            .drain(..)
            .tuples()
            .map(|(mut a, b)| {
                assert!(a.start >= b.start, "real assert 4a");
                a.merge(b);
                a
            })
            .collect();
    }

    #[inline]
    fn compress_take_first(&mut self, new_sample: Smoothness) {
        assert!(self.buffer.is_empty(), "real assert e2");
        assert!(fast_is_even(self.main.len()), "real assert e11");
        // println!("cmk compress_take_first");
        self.main
            .retain(|spaceline| new_sample.divides_u64(spaceline.time));
    }

    fn last(&self, step_index: u64, y_sample: Smoothness, y_smoothness: Smoothness) -> Spaceline {
        if self.buffer.is_empty() {
            // cmk00 would be nice to remove this clone
            return self.main.last().unwrap().clone();
        }
        // cmk00 in the special case in which the sample is 1 and the buffer is 1, can't we just return the buffer's item (as a ref???)

        let buffer_last = self.buffer.last().unwrap();
        let time = buffer_last.time;
        let start = buffer_last.start;
        let x_sample = buffer_last.sample;
        let last_inside_index = fast_mod(step_index, y_sample.as_u64());

        // cmk we have to clone because we compress in place (clone only half???)
        let mut buffer = self.buffer.clone();
        for inside_index in last_inside_index + 1..y_sample.as_u64() {
            let (_down_sample, down_step) = compute_sample_step(y_sample, y_smoothness);
            if !down_step.divides_u64(inside_index) {
                continue;
            }
            let inside_inside_index = down_step.divide_into(inside_index);

            let empty = Spaceline {
                sample: x_sample,
                start,
                pixels: vec![PIXEL_WHITE; buffer_last.pixels.len()],
                time: time + inside_index - last_inside_index,
                smoothness: buffer_last.smoothness,
            };
            Spacelines::push_internal(&mut buffer, inside_inside_index, empty);
        }
        assert!(buffer.len() == 1, "real assert b3");
        buffer.pop().unwrap()
    }

    fn push_internal(buffer: &mut Vec<Spaceline>, inside_index: u64, spaceline: Spaceline) {
        // Sampling & Averaging 3
        // We gather rows in a buffer until we have a full set of rows.
        // If we only wanted to keep one, we'd just push on inside_index==0

        if fast_is_even(inside_index) {
            buffer.push(spaceline);
        } else {
            let a = buffer.last_mut().unwrap();
            assert!(a.start >= spaceline.start, "cmk real assert 4b");
            a.merge(spaceline);
            let mut inside_inside = inside_index;
            loop {
                // shift inside_index to the right
                inside_inside >>= 1; // inside_inside /= 2;
                if fast_is_even(inside_inside) {
                    break;
                }
                let last = buffer.pop().unwrap();
                let a = buffer.last_mut().unwrap();
                assert!(a.start >= last.start, "cmk real assert 4c");
                a.merge(last);
            }
        }
    }

    #[inline]
    fn push(&mut self, inside_index: u64, spaceline: Spaceline) {
        Spacelines::push_internal(&mut self.buffer, inside_index, spaceline);
    }
}

pub struct SampledSpaceTime {
    step_index: u64,
    x_goal: u32,
    y_goal: u32,
    sample: Smoothness,
    spacelines: Spacelines,
    x_smoothness: Smoothness,
    y_smoothness: Smoothness,
}

/// Create a new in which you give the x_goal (space)
/// and the y_goal (time). The sample starts at 1 and
/// inner is a vector of one spaceline
impl SampledSpaceTime {
    pub fn new(
        x_goal: u32,
        y_goal: u32,
        x_smoothness: Smoothness,
        y_smoothness: Smoothness,
    ) -> Self {
        SampledSpaceTime {
            step_index: 0,
            x_goal,
            y_goal,
            sample: Smoothness::ONE,
            spacelines: Spacelines::new(x_smoothness),
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
                new_sample / self.sample == Smoothness::TWO,
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

    fn snapshot(&mut self, machine: &Machine) {
        self.step_index += 1;
        let inside_index = fast_mod(self.step_index, self.sample.as_u64());
        if inside_index == 0 {
            // We're starting a new set of spacelines, so flush the buffer and compress (if needed)
            self.spacelines.flush_buffer();
            self.compress_if_needed();
        }
        let (_down_sample, down_step) = compute_sample_step(self.sample, self.y_smoothness);
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

        self.spacelines.push(inside_inside_index, spaceline);
    }

    fn to_png(&self) -> Result<Vec<u8>, Error> {
        let last = self
            .spacelines
            .last(self.step_index, self.sample, self.y_smoothness);
        let x_sample: Smoothness = last.sample;
        let tape_width: u64 = last.pixels.len() as u64 * x_sample.as_u64();
        let tape_min_index = last.start;
        let x_actual: u32 = x_sample.divide_into(tape_width) as u32;
        let y_actual: u32 = self.spacelines.len() as u32;

        let row_bytes = x_actual;
        let mut packed_data = vec![0u8; row_bytes as usize * y_actual as usize];

        for y in 0..y_actual {
            let spaceline = &self.spacelines.get(y as usize, &last);
            let local_start = spaceline.start;
            let local_sample = spaceline.sample;
            let rel_sample = x_sample / local_sample;
            let row_start_byte_index: u32 = y * row_bytes;
            let x_start = int_div_ceil(local_start - tape_min_index, x_sample.as_u64() as i64);
            for x in x_start as u32..x_actual {
                let tape_index: i64 = x as i64 * x_sample.as_u64() as i64 + tape_min_index;
                // cmk consider changing asserts to debug_asserts
                assert!(
                    tape_index >= local_start,
                    "real assert if x_start is correct"
                );

                let local_spaceline_start: i64 =
                    (tape_index - local_start) / local_sample.as_u64() as i64;
                let slice = (local_spaceline_start
                    ..local_spaceline_start + rel_sample.as_u64() as i64)
                    .map(|i| {
                        spaceline
                            .pixels
                            .get(i as usize)
                            .copied()
                            .unwrap_or(PIXEL_WHITE)
                    })
                    .collect::<Vec<_>>();
                // cmk look at putting this back in
                // if local_spaceline_index >= spaceline.pixels.len() as i64 {
                //     break;
                // }

                // Sample & Averaging 5 --
                let value = Pixel::merge_slice_all(&slice, 0).0;
                if value != 0 {
                    let byte_index: u32 = x + row_start_byte_index;
                    packed_data[byte_index as usize] = value;
                }
            }
        }

        encode_png(x_actual, y_actual, &packed_data)
    }
}

#[inline]
fn int_div_ceil(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}
fn sample_rate(row: u64, goal: u32) -> Smoothness {
    let threshold = 2 * goal;
    let ratio = (row + 1) as f64 / threshold as f64;
    // For rows < threshold, ratio < 1, log2(ratio) is negative, and the ceil clamps to 0.
    let exponent = ratio.log2().ceil().max(0.0) as u8;
    Smoothness::new(exponent)
}

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
        for i in 0..256 {
            let g = 255 - ((255 - 165) * i / 255); // Green fades from 255 to 165
            let b = 255 - (255 * i / 255); // Blue fades from 255 to 0
            palette.extend_from_slice(&[255, g as u8, b as u8]);
        }

        // Set the palette before writing the header
        encoder.set_palette(palette);

        let mut writer = encoder.write_header().map_err(|_| Error::EncodingError)?;
        writer
            .write_image_data(image_data)
            .map_err(|_| Error::EncodingError)?;
    }
    Ok(buf)
}

#[wasm_bindgen]
pub struct SpaceTimeMachine {
    machine: Machine,
    space_time: SampledSpaceTime,
}

// impl iterator for spacetime machine
impl Iterator for SpaceTimeMachine {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        self.machine.next()?;
        self.space_time.snapshot(&self.machine);
        Some(())
    }
}

#[wasm_bindgen]
impl SpaceTimeMachine {
    #[wasm_bindgen(constructor)]
    pub fn from_str(
        s: &str,
        goal_x: u32,
        goal_y: u32,
        x_smoothness: u8,
        y_smoothness: u8,
    ) -> Result<SpaceTimeMachine, String> {
        Ok(SpaceTimeMachine {
            machine: Machine::from_string(s)?,
            space_time: SampledSpaceTime::new(
                goal_x,
                goal_y,
                Smoothness(x_smoothness),
                Smoothness(y_smoothness),
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

    #[wasm_bindgen]
    pub fn png_data(&self) -> Vec<u8> {
        self.space_time
            .to_png()
            .unwrap_or_else(|e| format!("{:?}", e).into_bytes())
    }

    #[wasm_bindgen]
    pub fn step_count(&self) -> u64 {
        self.space_time.step_index + 1
    }

    #[wasm_bindgen]
    pub fn count_ones(&self) -> u32 {
        self.machine.count_ones()
    }

    #[wasm_bindgen]
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
    pub fn new(max_value: u64, total_frames: u32) -> Self {
        // if total_frames < 2 {
        //     panic!("Number of frames must be at least 2.");
        // }
        Self {
            current_frame: 0,
            total_frames,
            max_value,
        }
    }
}

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

#[inline(always)]
fn fast_div<T>(numerator: T, denominator: T) -> T
where
    T: PrimInt + Shr<u32, Output = T>, // Ensures integer behavior & right shift
{
    debug_assert!(
        (denominator & (denominator - T::one())) == T::zero(),
        "denominator must be a power of two"
    );

    numerator >> denominator.trailing_zeros()
}

#[inline(always)]
fn fast_mod<T>(x: T, sample: T) -> T
where
    T: Copy + std::ops::BitAnd<Output = T> + std::ops::Sub<Output = T> + From<u8> + PartialEq,
{
    debug_assert!(
        sample & (sample - T::from(1)) == T::from(0),
        "sample must be a power of two"
    );
    x & (sample - T::from(1))
}

#[inline(always)]
fn fast_is_even<T>(x: T) -> bool
where
    T: Copy + std::ops::BitAnd<Output = T> + std::ops::Sub<Output = T> + From<u8> + PartialEq,
{
    (x & T::from(1)) == T::from(0)
}

#[inline(always)]
fn fast_rem_euclid(dividend: i64, divisor: u64) -> i64 {
    debug_assert!(divisor.is_power_of_two(), "divisor must be a power of two");
    let mask = (divisor - 1) as i64;
    let remainder = dividend & mask;
    remainder + ((divisor as i64) & (remainder >> 63))
}

/// The function computes
/// ```text
/// sample_out = min(sample, smoothness)
/// step = sample / sample_out
/// ```
#[inline(always)]
fn compute_sample_step(sample: Smoothness, smoothness: Smoothness) -> (Smoothness, Smoothness) {
    let is_down_sample = (sample > smoothness) as u64; // 1 if true, 0 if false

    // Prevent underflow by using saturating_sub, which ensures non-negative results
    let shift_amount = sample
        .as_u64()
        .trailing_zeros()
        .saturating_sub(smoothness.as_u64().trailing_zeros()); // cmk0000000000

    let sample_out =
        Smoothness::from_u64(sample.as_u64() >> (shift_amount * is_down_sample as u32)); // Shift only if down sampling
    let step = sample / sample_out;

    // ✅ Debug assertions to validate correctness
    debug_assert!(
        sample_out <= smoothness,
        "sample_out must not exceed smoothness"
    );
    debug_assert!(
        step.as_u64() * sample_out.as_u64() == sample.as_u64(),
        "step * sample_out must equal sample"
    );

    debug_assert!(
        sample_out == sample.min(smoothness),
        "sample_out must be min(sample, smoothness)"
    );
    debug_assert!(
        step == sample / sample_out,
        "step must be sample / sample_out"
    );

    (sample_out, step)
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Smoothness(u8);

impl std::ops::Div for Smoothness {
    type Output = Self;

    /// Will always be at least 1.
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.0 >= rhs.0,
            "Divisor must be less than or equal to dividend"
        );
        // Subtract exponents; if the subtrahend is larger, saturate to 0.
        Smoothness(self.0.saturating_sub(rhs.0))
    }
}

// cmk make the auto constructor so private that it can't be used w/ modules, so that new check is run.

impl Smoothness {
    /// The smallest valid `Smoothness` value, representing `2^0 = 1`.
    pub const ONE: Self = Smoothness(0);
    pub const TWO: Self = Smoothness(1);

    #[inline]
    pub fn new(value: u8) -> Self {
        debug_assert!(value <= 63, "Value must be 63 or less");
        Smoothness(value)
    }

    #[inline]
    pub fn as_u64(self) -> u64 {
        1 << self.0
    }

    // from u64
    pub fn from_u64(value: u64) -> Self {
        assert!(value.is_power_of_two(), "Value must be a power of two");
        Smoothness::new(value.trailing_zeros() as u8)
    }

    #[inline]
    pub fn log2(self) -> u8 {
        self.0
    }

    #[inline(always)]
    fn rem_into<T>(self, x: T) -> T
    where
        T: Copy + std::ops::BitAnd<Output = T> + std::ops::Sub<Output = T> + From<u64> + PartialEq,
    {
        x & (T::from(self.as_u64()) - T::from(1u64))
    }

    #[inline(always)]
    fn divide_into<T>(self, x: T) -> T
    where
        T: Copy + std::ops::Shr<u8, Output = T>,
    {
        x >> self.0
    }

    #[inline(always)]
    pub fn divides_u64(self, x: u64) -> bool {
        // If x is divisible by 2^(self.0), shifting right then left recovers x.
        (x >> self.0) << self.0 == x
    }

    #[inline(always)]
    pub fn divides_i64(self, x: i64) -> bool {
        (x >> self.0) << self.0 == x
    }

    /// Returns `true` if `self` divides `other`, i.e. if 2^(self.0) divides 2^(other.0).
    #[inline(always)]
    pub fn divides_smoothness(self, other: Smoothness) -> bool {
        self.0 <= other.0
    }
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

    /// See https://en.wikipedia.org/wiki/Busy_beaver
    #[test]
    fn bb5_champ_space_time_native() -> Result<(), Error> {
        let mut machine: Machine = BB5_CHAMP.parse()?; // cmk
                                                       // let mut machine: Machine = BB6_CONTENDER.parse()?;

        let goal_x: u32 = 1000;
        let goal_y: u32 = 1000;
        let x_smoothness: Smoothness = Smoothness::ONE;
        let y_smoothness: Smoothness = Smoothness::ONE;
        let mut sample_space_time =
            SampledSpaceTime::new(goal_x, goal_y, x_smoothness, y_smoothness);

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
        fs::write("test.png", &png_data).unwrap(); // cmk handle error

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
        let s = BB5_CHAMP;
        let goal_x: u32 = 1000;
        let goal_y: u32 = 1000;
        let x_smoothness: Smoothness = Smoothness::ONE;
        let y_smoothness: Smoothness = Smoothness::ONE;
        let n = 1_000_000;
        let mut space_time_machine = SpaceTimeMachine::from_str(
            s,
            goal_x,
            goal_y,
            x_smoothness.log2(),
            y_smoothness.log2(),
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
        fs::write("test_js.png", &png_data).map_err(|e| e.to_string())?;

        assert_eq!(space_time_machine.space_time.step_index + 1, 47_176_870);
        assert_eq!(space_time_machine.machine.count_ones(), 4098);
        assert_eq!(space_time_machine.machine.state, 7);
        assert_eq!(space_time_machine.machine.tape_index, -12242);

        Ok(())
    }

    #[wasm_bindgen_test]
    #[test]
    fn machine_7_135_505() -> Result<(), Error> {
        let _machine_a: Machine = Machine_7_135_505_A.parse()?;
        let _machine_b: Machine = Machine_7_135_505_B.parse()?;
        Ok(())
    }

    // Create a test that runs bb5 champ to halting and then prints the time it took
    // to run the test
    #[wasm_bindgen_test]
    #[test]
    fn bb5_champ_time() -> Result<(), String> {
        let start = std::time::Instant::now();
        let step_count = 1 + BB5_CHAMP.parse::<Machine>().unwrap().count();
        let duration = start.elapsed();
        println!(
            "Steps: {}, Duration: {:?}",
            step_count.separate_with_commas(),
            duration
        );
        assert_eq!(step_count, 47_176_870);
        Ok(())
    }
}
