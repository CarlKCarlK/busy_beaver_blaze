use core::fmt;
use derive_more::derive::Display;
use derive_more::Error as DeriveError;
use png::{BitDepth, ColorType, Encoder};
use std::{
    ops::{Index, IndexMut},
    str::FromStr,
};
use thousands::Separable;
use wasm_bindgen::prelude::*;

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

    fn index_range_to_string(&self, range: std::ops::RangeInclusive<i32>) -> String {
        let mut s = String::new();
        for i in range {
            s.push_str(&self[i].to_string());
        }
        s
    }

    #[inline]
    pub fn min_index(&self) -> i32 {
        -(self.negative.len() as i32)
    }

    #[inline]
    pub fn max_index(&self) -> i32 {
        self.nonnegative.len() as i32 - 1
    }

    #[inline]
    pub fn width(&self) -> u32 {
        (self.max_index() - self.min_index() + 1) as u32
    }
}

#[wasm_bindgen]
struct Machine {
    state: u8,
    tape_index: i32,
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
    pub fn space_time_js(&mut self, goal_x: u32, goal_y: u32, early_stop: Option<u32>) -> Vec<u8> {
        let mut sampled_space_time = SampledSpaceTime::new(goal_x, goal_y);

        while self.next().is_some()
            && early_stop.is_none_or(|early_stop| sampled_space_time.step_count < early_stop)
        {
            sampled_space_time.snapshot(&self);
        }

        sampled_space_time
            .to_png()
            .unwrap_or_else(|e| format!("{:?}", e).into_bytes())
    }

    #[wasm_bindgen]
    pub fn count_js(&mut self, early_stop: Option<u32>) -> u32 {
        let mut step_count = 0;

        while self.next().is_some() && early_stop.is_none_or(|early_stop| step_count < early_stop) {
            step_count += 1;
        }

        step_count
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
        let input = self.tape[self.tape_index];
        let per_input = &program.inner[self.state as usize][input as usize];
        self.tape[self.tape_index] = per_input.next_symbol;
        self.tape_index += per_input.direction as i32;
        self.state = per_input.next_state;
        if (self.state as usize) < program.state_count {
            Some(())
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct Program {
    state_count: usize,
    symbol_count: usize,
    inner: Vec<Vec<Action>>,
}

// impl a parser for the strings like this
// "   A	B	C	D	E
// 0	1RB	1RC	1RD	1LA	1RH
// 1	1LC	1RB	0LE	1LD	0LA"
impl FromStr for Program {
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
        let mut vec_of_vec: Vec<Vec<Action>> = lines
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
                    })
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
        let mut inner: Vec<Vec<Action>> = (0..state_count)
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
                inner[i].push(item); // Move item into transposed[i]
            }
        }

        Ok(Program {
            state_count,
            symbol_count,
            inner,
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
        let mut step_count = 0;

        loop {
            if step_count % debug_interval == 0 {
                println!("Step {}: {:?}", step_count.separate_with_commas(), self);
            }
            if self.next().is_none() {
                break;
            }
            step_count += 1;
        }

        step_count
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

    #[display("Invalid encoding encountered")]
    EncodingError,
}

// Implement conversions manually where needed
impl From<std::num::ParseIntError> for Error {
    fn from(err: std::num::ParseIntError) -> Self {
        Error::ParseIntError(err)
    }
}

struct Spaceline {
    sample: u32,
    start: i32,
    values: Vec<u8>,
    time: u32,
}

impl Default for Spaceline {
    fn default() -> Self {
        Spaceline {
            sample: 1,
            start: 0,
            values: vec![0; 1],
            time: 0,
        }
    }
}

struct SampledSpaceTime {
    step_count: u32,
    x_goal: u32,
    y_goal: u32,
    sample: u32,
    spacelines: Vec<Spaceline>,
}

/// Create a new in which you give the x_goal (space)
/// and the y_goal (time). The sample starts at 1 and
/// inner is a vector of one spaceline

impl SampledSpaceTime {
    fn new(x_goal: u32, y_goal: u32) -> Self {
        SampledSpaceTime {
            step_count: 0,
            x_goal,
            y_goal,
            sample: 1,
            spacelines: vec![Spaceline::default()],
        }
    }

    fn compress_if_needed(&mut self) {
        let new_sample = sample_rate(self.step_count, self.y_goal);
        if new_sample != self.sample {
            self.spacelines
                .retain(|spaceline| spaceline.time % new_sample == 0);
            self.sample = new_sample;
        }
    }

    fn snapshot(&mut self, machine: &Machine) {
        self.step_count += 1;

        if self.step_count % self.sample != 0 {
            return;
        }
        self.compress_if_needed();

        let tape = &machine.tape;
        let tape_width = tape.width();
        let tape_min_index = tape.min_index();
        let tape_max_index = tape.max_index();
        let x_sample = sample_rate(tape_width, self.x_goal);

        let sample_start: i32 = tape_min_index
            + ((x_sample as i32 - tape_min_index.rem_euclid(x_sample as i32)) % x_sample as i32);

        debug_assert!(
            sample_start >= tape_min_index
                && sample_start % x_sample as i32 == 0
                && sample_start - tape_min_index < x_sample as i32
        );

        let mut values = Vec::with_capacity(self.x_goal as usize * 2);
        for sample_index in (sample_start..=tape_max_index).step_by(x_sample as usize) {
            values.push(tape[sample_index]);
        }

        self.spacelines.push(Spaceline {
            sample: x_sample,
            start: sample_start,
            values,
            time: self.step_count,
        });
    }

    fn to_png(&self) -> Result<Vec<u8>, Error> {
        let last = self.spacelines.last().unwrap(); // Safe because we always have at least one
        let x_sample: u32 = last.sample;
        let tape_width: u32 = last.values.len() as u32 * x_sample;
        let tape_min_index = last.start;
        let x_actual: u32 = tape_width / x_sample;
        let y_actual = self.spacelines.len() as u32;

        let row_bytes = ((x_actual as usize) + 7) / 8;
        let mut packed_data = vec![0u8; row_bytes * (y_actual as usize)];

        // First row is always empty, so start at 1
        for y in 1..y_actual {
            let spaceline = &self.spacelines[y as usize];
            let local_start = spaceline.start;
            let local_sample = spaceline.sample;
            let row_start_byte_index: u32 = y * row_bytes as u32;
            let x_start = int_div_ceil(local_start - tape_min_index, x_sample as i32);

            for x in x_start as u32..x_actual {
                let tape_index: i32 = (x * x_sample) as i32 + tape_min_index;
                debug_assert!(
                    tape_index >= local_start,
                    "real assert if x_start is correct"
                );

                let local_spaceline_index: i32 = (tape_index - local_start) / local_sample as i32;
                if local_spaceline_index >= spaceline.values.len() as i32 {
                    break;
                }

                let value = spaceline.values[local_spaceline_index as usize];
                if value != 0 {
                    debug_assert!(value == 1);
                    let bit_index = 7 - (x % 8); // PNG is backwards
                    let byte_index: u32 = x / 8 + row_start_byte_index;
                    packed_data[byte_index as usize] |= 1 << bit_index;
                }
            }
        }

        encode_png(x_actual, y_actual, &packed_data)
    }
}

#[inline]
fn int_div_ceil(a: i32, b: i32) -> i32 {
    (a + b - 1) / b
}
fn sample_rate(row: u32, goal: u32) -> u32 {
    let threshold = 2 * goal;
    let ratio = (row + 1) as f64 / threshold as f64;
    // For rows < threshold, ratio < 1, log2(ratio) is negative, and the ceil clamps to 0.
    let exponent = ratio.log2().ceil().max(0.0) as u32;
    2_u32.pow(exponent)
}

fn encode_png(width: u32, height: u32, image_data: &[u8]) -> Result<Vec<u8>, Error> {
    // assert!(image_data.len() == (width * height) as usize);
    println!("cmk {:?}, {width}x{height}", image_data.len());
    let mut buf = Vec::new();
    {
        // Create an encoder that writes directly into `buf`
        let mut encoder = Encoder::new(&mut buf, width, height);
        encoder.set_color(ColorType::Indexed);
        encoder.set_depth(BitDepth::One);
        encoder.set_palette(vec![
            255, 255, 255, // White (Background, 0)
            255, 165, 0, // Orange (Foreground, 1)
        ]);

        // Instead of using the stream writer, get a writer that can encode
        // the entire image data in one go.
        let mut writer = encoder.write_header().map_err(|_| Error::EncodingError)?;
        // This method writes all the image data at once.
        let result = writer.write_image_data(image_data);
        println!("cmk {:?}", result);
        result.map_err(|_| Error::EncodingError)?; // cmk define a From
    }
    // At this point, `buf` contains the PNG data.
    Ok(buf)
}

// tests
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

        let early_stop = Some(10_000_000);
        let step_count = machine.count_js(early_stop);

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

    // cmk: really need to understand i32/u32/usize/isize/u64/i64
    // cmk: Bug -- displays an extra time step
    // cmk00: -- y sampling is working, but x sampling is not

    /// See https://en.wikipedia.org/wiki/Busy_beaver
    #[test]
    fn bb5_champ_space_time() -> Result<(), Error> {
        let mut machine: Machine = BB5_CHAMP.parse()?; // cmk
                                                       // let mut machine: Machine = BB6_CONTENDER.parse()?;

        let goal_x: u32 = 1000;
        let goal_y: u32 = 1000;
        let mut sample_space_time = SampledSpaceTime::new(goal_x, goal_y);

        let early_stop = Some(10_500_000);
        let debug_interval = Some(1_000_000);

        while machine.next().is_some()
            && early_stop.is_none_or(|early_stop| sample_space_time.step_count < early_stop)
        {
            if debug_interval
                .is_none_or(|debug_interval| sample_space_time.step_count % debug_interval == 0)
            {
                println!(
                    "Step {}: {:?},\t{}",
                    sample_space_time.step_count.separate_with_commas(),
                    machine,
                    machine.tape.index_range_to_string(-10..=10)
                );
            }

            sample_space_time.snapshot(&machine);
        }

        let png_data = sample_space_time.to_png()?;
        fs::write("test.png", &png_data).unwrap(); // cmk handle error

        println!(
            "Final: Steps {}: {:?}, #1's {}",
            sample_space_time.step_count.separate_with_commas(),
            machine,
            machine.tape.count_ones()
        );

        if early_stop.is_none() {
            assert_eq!(sample_space_time.step_count, 47_176_870);
            assert_eq!(machine.tape.count_ones(), 4098);
            assert_eq!(machine.state, 7);
            assert_eq!(machine.tape_index, -12242);
        }

        Ok(())
    }
}
