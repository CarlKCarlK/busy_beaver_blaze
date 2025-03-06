use instant::Instant;
use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use wasm_bindgen::prelude::*;

use crate::{
    Machine, PixelPolicy, is_even, sample_rate, space_by_time::SpaceByTime, spacelines::Spacelines,
};

#[wasm_bindgen]
pub struct SpaceByTimeMachine {
    machine: Machine,
    pub(crate) space_by_time: SpaceByTime,
}

// impl iterator for spacetime machine
#[allow(clippy::missing_trait_methods)]
impl Iterator for SpaceByTimeMachine {
    type Item = ();

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let previous_tape_index = self.machine.next()?;
        self.space_by_time
            .snapshot(&self.machine, previous_tape_index);
        Some(())
    }
}

#[wasm_bindgen]
#[allow(clippy::min_ident_chars)]
impl SpaceByTimeMachine {
    #[wasm_bindgen(constructor)]
    pub fn from_str(
        program: &str,
        goal_x: u32,
        goal_y: u32,
        binning: bool,
        skip: u64,
    ) -> Result<Self, String> {
        let mut machine = Machine::from_string(program)?;
        for _ in 0..skip {
            if machine.next().is_none() {
                return Err("Machine halted while skipping".to_owned());
            }
        }
        let space_by_time = SpaceByTime::new_skipped(
            machine.tape(),
            skip,
            goal_x,
            goal_y,
            if binning {
                PixelPolicy::Binning
            } else {
                PixelPolicy::Sampling
            },
        );
        Ok(Self {
            machine,
            space_by_time,
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
    #[inline]
    #[must_use]
    pub fn png_data(&mut self) -> Vec<u8> {
        self.space_by_time
            .to_png()
            .unwrap_or_else(|e| format!("{e:?}").into_bytes())
    }

    #[wasm_bindgen]
    #[inline]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn step_count(&self) -> u64 {
        self.space_by_time.step_index() + 1
    }

    #[wasm_bindgen]
    #[inline]
    #[must_use]
    pub fn count_ones(&self) -> u32 {
        self.machine.count_ones()
    }

    #[wasm_bindgen]
    #[inline]
    #[must_use]
    pub fn is_halted(&self) -> bool {
        self.machine.is_halted()
    }
}

#[allow(clippy::too_many_lines, clippy::missing_panics_doc)]
impl SpaceByTimeMachine {
    #[must_use]
    pub fn from_str_in_parts(
        max_rows: u64,
        part_count: u64,
        program_string: &str,
        goal_x: u32,
        goal_y: u32,
        binning: bool,
    ) -> Self {
        assert!(max_rows > 0); // panic if early_stop is 0
        assert!(part_count > 0); // panic if part_count is 0
        let mut rows_per_part = max_rows.div_ceil(part_count);

        let y_stride = sample_rate(rows_per_part, goal_y);
        rows_per_part += y_stride.offset_to_align(rows_per_part as usize) as u64;
        // assert_eq!(y_stride.double(), sample_rate(rows_per_part, goal_y), "+1?");
        assert!(y_stride.divides_u64(rows_per_part), "even?");

        println!("Part max_rows_per_part: {rows_per_part}");
        let range_list: Vec<_> = (0..max_rows)
            .step_by(rows_per_part as usize)
            .map(|start| start..(start + rows_per_part).min(max_rows))
            .collect();

        let results: Vec<(bool, Self)> = range_list
            .par_iter()
            // .iter() // cmk000000000000
            .enumerate()
            .map(|(part_index, range)| {
                let (start, end) = (range.start, range.end);
                println!("{part_index}: Start: {start}, End: {end}");
                let mut space_by_time_machine =
                    Self::from_str(program_string, goal_x, goal_y, binning, start)
                        .expect("Failed to create machine");
                for _time in start + 1..end {
                    if space_by_time_machine.next().is_none() {
                        break;
                    }
                }

                let space_by_time = &mut space_by_time_machine.space_by_time;
                let inside_index = space_by_time
                    .y_stride
                    .rem_into_u64(space_by_time.step_index() + 1);
                // This should be 0 on all but the last part
                if (part_index as u64) < part_count - 1 {
                    assert_eq!(inside_index, 0, "real assert 1");
                }
                if inside_index == 0 {
                    // We're starting a new set of spacelines, so flush the buffer and compress (if needed)
                    space_by_time.spacelines.flush_buffer0();
                    space_by_time.compress_if_needed();
                }

                (true, space_by_time_machine)
            })
            .collect();

        let mut results_iter = results.into_iter();
        let (_continues0, mut space_by_time_machine_first) = results_iter.next().unwrap();
        let space_by_time_first = &mut space_by_time_machine_first.space_by_time;
        assert!(
            space_by_time_first.spacelines.buffer0.is_empty() || part_count == 1,
            "real assert 3"
        );

        let mut index: usize = 0;
        for (_continues, space_by_time_machine) in results_iter {
            index += 1;
            let space_by_time = space_by_time_machine.space_by_time;
            let spacelines = space_by_time.spacelines;
            let main = spacelines.main;
            let buffer0 = spacelines.buffer0;
            if index < part_count as usize - 1 {
                assert!(buffer0.is_empty(), "real assert 2");
            }
            for spaceline in main {
                space_by_time_first.spacelines.main.push(spaceline);
            }
            for (spaceline, weight) in buffer0 {
                space_by_time_first
                    .spacelines
                    .buffer0
                    .push((spaceline, weight));
            }
        }

        loop {
            let len = space_by_time_first.spacelines.len();
            assert!(len > 0, "real assert 5");
            if len < goal_y as usize * 2 {
                break;
            }
            println!("len: {len} is too long, compressing...");
            if !is_even(len) {
                let last = space_by_time_first.spacelines.main.pop().unwrap();
                space_by_time_first
                    .spacelines
                    .buffer0
                    .insert(0, (last, space_by_time_first.y_stride));
            }
            let len2 = space_by_time_first.spacelines.len();
            assert!(len2 <= len);
            assert!(is_even(len2), "real assert 6");
            if binning {
                space_by_time_first.spacelines.main = space_by_time_first
                    .spacelines
                    .main
                    .drain(..)
                    .tuples()
                    .map(|(mut a, b)| {
                        assert!(a.tape_start() >= b.tape_start(), "real assert 4a");
                        a.merge(&b);
                        a
                    })
                    .collect();
            } else {
                // cmk000000 buggy
                let new_stride = space_by_time_first.y_stride.double();
                println!("new_stride: {new_stride:?}");
                let mut expect = true;
                for spaceline in &space_by_time_first.spacelines.main {
                    let divides = new_stride.divides_u64(spaceline.time);
                    assert_eq!(divides, expect, "real assert 7");
                    expect = !expect;
                }
                // assert!(expect, "real assert 8");
                space_by_time_first
                    .spacelines
                    .main
                    .retain(|spaceline| new_stride.divides_u64(spaceline.time));
            }
            println!("new len: {}", space_by_time_first.spacelines.len());
            // assert!(space_by_time_first.spacelines.len() * 2 <= len);
        }

        println!("main's y_stride {:?}", &space_by_time_first.y_stride);
        for (spaceline, weight) in &space_by_time_first.spacelines.buffer0 {
            println!("time: {}, weight: {weight:?}", spaceline.time);
        }

        // take the buffer0 leaving it empty
        let buffer_old = core::mem::take(&mut space_by_time_first.spacelines.buffer0);
        let buffer0 = &mut space_by_time_first.spacelines.buffer0;
        let mut old_weight = None;
        for (spaceline, weight) in buffer_old {
            assert!(
                old_weight.is_none() || old_weight.unwrap() >= weight,
                "should be monotonic"
            );
            assert!(
                weight <= space_by_time_first.y_stride,
                "should be <= y_stride"
            );
            if weight == space_by_time_first.y_stride {
                // This is a special case where we have a spaceline that is exactly the y_stride, so we can just push it to the main buffer
                space_by_time_first.spacelines.main.push(spaceline);
                continue;
            }
            Spacelines::push_internal(buffer0, spaceline, weight);
            space_by_time_first.step_index += weight.as_u64();
            old_weight = Some(weight);
        }
        println!("again main's y_stride {:?}", &space_by_time_first.y_stride);
        for (spaceline, weight) in &space_by_time_first.spacelines.buffer0 {
            println!("again: time: {}, weight: {weight:?}", spaceline.time);
        }

        space_by_time_machine_first
    }

    #[inline]
    #[must_use]
    pub const fn machine(&self) -> &Machine {
        &self.machine
    }

    #[inline]
    #[must_use]
    pub const fn step_index(&self) -> u64 {
        self.space_by_time.step_index()
    }

    #[inline]
    #[must_use]
    pub fn state(&self) -> u8 {
        self.machine.state()
    }

    #[inline]
    #[must_use]
    pub fn tape_index(&self) -> i64 {
        self.machine.tape_index()
    }
}
