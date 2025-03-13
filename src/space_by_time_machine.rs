use core::ops::Range;
use std::{collections::HashMap, fs};

use instant::Instant;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use wasm_bindgen::prelude::*;

use crate::{
    Machine, PixelPolicy, PowerOfTwo, Snapshot, sample_rate, space_by_time::SpaceByTime,
    spacelines::Spacelines,
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
            .to_png(
                self.machine.tape.negative.len(),
                self.machine.tape.nonnegative.len(),
                self.space_by_time.x_goal as usize,
                self.space_by_time.y_goal as usize,
            )
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

#[allow(
    clippy::too_many_lines,
    clippy::missing_panics_doc,
    clippy::shadow_reuse
)]
impl SpaceByTimeMachine {
    #[inline]
    #[must_use]
    pub fn png_data_and_packed_data(&mut self) -> (Vec<u8>, u32, u32, Vec<u8>) {
        self.space_by_time
            .to_png_and_packed_data(
                self.machine.tape.negative.len(),
                self.machine.tape.nonnegative.len(),
                self.space_by_time.x_goal as usize,
                self.space_by_time.y_goal as usize,
            )
            .unwrap()
    }

    #[must_use]
    pub fn from_str_in_parts(
        // cmk change the order of the inputs
        early_stop: u64,
        part_count_goal: u64,
        program_string: &str,
        goal_x: u32,
        goal_y: u32,
        binning: bool,
        frame_step_indexes: &[u64],
    ) -> Self {
        let snapshots_and_space_by_time_machines = Self::run_parts(
            early_stop,
            part_count_goal,
            program_string,
            goal_x,
            goal_y,
            binning,
            frame_step_indexes,
        );

        Self::combine_results(goal_x, goal_y, snapshots_and_space_by_time_machines)
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

    fn create_range_list(early_stop: u64, part_count_goal: u64, goal_y: u32) -> Vec<Range<u64>> {
        let mut rows_per_part = early_stop.div_ceil(part_count_goal);

        let y_stride = sample_rate(rows_per_part, goal_y);
        rows_per_part += y_stride.offset_to_align(rows_per_part as usize) as u64;
        assert!(y_stride.divides_u64(rows_per_part), "even?");

        (0..early_stop)
            .step_by(rows_per_part as usize)
            .map(|start| start..(start + rows_per_part).min(early_stop))
            .collect()
    }

    // cmk000 need to handle the case of early halting.
    fn run_parts(
        early_stop: u64,
        part_count_goal: u64,
        program_string: &str,
        goal_x: u32,
        goal_y: u32,
        binning: bool,
        frame_step_indexes: &[u64],
    ) -> Vec<(Vec<Snapshot>, Self)> {
        assert!(early_stop > 0); // panic if early_stop is 0
        assert!(part_count_goal > 0); // panic if part_count_goal is 0

        let range_list = Self::create_range_list(early_stop, part_count_goal, goal_y);
        let part_count = range_list.len() as u64;

        range_list
            .par_iter()
            // .iter()
            .enumerate()
            .map(|(part_index, range)| {
                let (start, end) = (range.start, range.end);

                // Create a hashmap where the key is any frame_step_indexes between start..end and
                // the value is a list of all the enumeration index in the frame_step_indexes vector.
                let mut step_index_to_frame_index: HashMap<u64, Vec<usize>> = HashMap::new();
                for (index, &step_index) in frame_step_indexes.iter().enumerate() {
                    if step_index >= start && step_index < end {
                        step_index_to_frame_index
                            .entry(step_index)
                            .or_default()
                            .push(index);
                    }
                }

                let mut space_by_time_machine =
                    Self::from_str(program_string, goal_x, goal_y, binning, start)
                        .expect("Failed to create machine");

                let mut snapshots: Vec<Snapshot> = vec![];
                for step_index in start + 1..end {
                    if let Some(frame_indexes) = step_index_to_frame_index.get(&step_index) {
                        snapshots
                            .push(Snapshot::new(frame_indexes.clone(), &space_by_time_machine));
                    }
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
                    space_by_time.compress_cmk1_y_if_needed();
                }

                (snapshots, space_by_time_machine)
            })
            .collect()
    }

    // cmk is it sometimes x_goal and sometimes goal_x????
    fn combine_results(
        x_goal: u32,
        y_goal: u32,
        snapshots_and_space_by_time_machines: Vec<(Vec<Snapshot>, Self)>,
    ) -> Self {
        let part_count = snapshots_and_space_by_time_machines.len() as u64;
        // Extract FIRST result
        let mut results_iter = snapshots_and_space_by_time_machines.into_iter();
        let result = results_iter.next().unwrap();
        let (snapshots_first, mut space_by_time_machine_first) = result;
        let space_by_time_first = &mut space_by_time_machine_first.space_by_time;
        assert!(space_by_time_first.spacelines.buffer0.is_empty() || part_count == 1,);

        for mut snapshot_first in snapshots_first {
            let png_data = snapshot_first.to_png(x_goal, y_goal).unwrap(); //cmk0000 need to handle
            for frame_index in &snapshot_first.frame_indexes {
                let cmk_file = format!(r"M:\deldir\bb\frames_test\cmk{frame_index:07}.png");
                fs::write(cmk_file, &png_data).unwrap();
            }
        }

        let mut index: usize = 0;
        for (snapshots_other, space_by_time_machine_other) in results_iter {
            index += 1;
            let space_by_time_other = space_by_time_machine_other.space_by_time;
            Self::assert_empty_buffer_if_not_last_part(&space_by_time_other, index, part_count);

            for mut snapshot_other in snapshots_other {
                // cmk this is convoluted way to combine these two
                let mut space_by_time_combo = space_by_time_first.clone();
                Self::combine2(&mut space_by_time_combo, snapshot_other.space_by_time);

                // cmk000000000 make this a function
                loop {
                    space_by_time_combo.compress_cmk3_y_if_needed(y_goal, None);
                    space_by_time_combo.reduce_buffer0();

                    if space_by_time_combo.spacelines.len() <= y_goal as usize * 2 {
                        break;
                    }
                }

                snapshot_other.space_by_time = space_by_time_combo;
                let png_data = snapshot_other.to_png(x_goal, y_goal).unwrap(); //cmk0000 need to handle
                for frame_index in &snapshot_other.frame_indexes {
                    let cmk_file = format!(r"M:\deldir\bb\frames_test\cmk{frame_index:07}.png");
                    fs::write(cmk_file, &png_data).unwrap();
                }
            }

            Self::combine2(space_by_time_first, space_by_time_other);
            space_by_time_machine_first.machine = space_by_time_machine_other.machine;
        }

        // cmk000000000 make this a function
        loop {
            space_by_time_first.compress_cmk3_y_if_needed(y_goal, None);
            space_by_time_first.reduce_buffer0();

            if space_by_time_first.spacelines.len() <= y_goal as usize * 2 {
                break;
            }
        }

        space_by_time_machine_first
    }

    #[inline]
    fn assert_empty_buffer_if_not_last_part(
        space_by_time_other: &SpaceByTime,
        index: usize,
        part_count: u64,
    ) {
        if index < part_count as usize - 1 {
            assert!(
                space_by_time_other.spacelines.buffer0.is_empty(),
                "real assert 2"
            );
        }
    }

    // cmk move to SpaceByTime
    fn combine2(space_by_time_first: &mut SpaceByTime, space_by_time_other: SpaceByTime) {
        let y_stride = space_by_time_first.y_stride;
        let spacelines_other = space_by_time_other.spacelines;
        let main_other = spacelines_other.main;
        let buffer0_other = spacelines_other.buffer0;

        // If y_strides match, add other's main spacelines to the main buffer else add to the buffer0
        let (mut previous_y_stride, mut previous_time) = if space_by_time_other.y_stride == y_stride
        {
            for spaceline in main_other {
                space_by_time_first.spacelines.main.push(spaceline);
            }
            (
                space_by_time_first.y_stride,
                space_by_time_first.spacelines.main.last().unwrap().time,
            )
        } else {
            for spaceline in main_other {
                space_by_time_first
                    .spacelines
                    .buffer0
                    .push((spaceline, space_by_time_other.y_stride));
            }
            (
                space_by_time_other.y_stride,
                space_by_time_first
                    .spacelines
                    .buffer0
                    .last()
                    .unwrap()
                    .0
                    .time,
            )
        };

        // Add other's buffer0 spacelines to buffer0
        for (spaceline, weight) in buffer0_other {
            let time = spaceline.time;
            assert!(
                time == previous_time + previous_y_stride.as_u64(),
                "mind the gap"
            );
            previous_y_stride = weight;
            previous_time = time;
            space_by_time_first
                .spacelines
                .buffer0
                .push((spaceline, weight));
        }
    }

    #[inline]
    #[allow(unused_variables)]
    pub(crate) fn audit_one(
        space_by_time: &SpaceByTime,
        previous_y_stride: Option<PowerOfTwo>,
        previous_time: Option<u64>,
        early_stop: Option<u64>,
        binning: bool,
    ) {
        // on debug compiles call audit_one_internal otherwise do nothing
        #[cfg(debug_assertions)]
        Self::audit_one_internal(
            space_by_time,
            previous_y_stride,
            previous_time,
            early_stop,
            binning,
        );
    }

    #[cfg(debug_assertions)]
    fn audit_one_internal(
        space_by_time: &SpaceByTime,
        mut previous_y_stride: Option<PowerOfTwo>,
        mut previous_time: Option<u64>,
        early_stop: Option<u64>,
        binning: bool,
    ) {
        let y_stride = space_by_time.y_stride;
        if let Some(previous_y_stride) = previous_y_stride {
            assert_eq!(
                y_stride, previous_y_stride,
                "from part to part, the stride should be the same"
            );
        }
        let spacelines = &space_by_time.spacelines;
        let main = &spacelines.main;
        for spaceline in main {
            if let Some(previous_time) = previous_time {
                assert!(
                    spaceline.time == previous_time + y_stride.as_u64(),
                    "mind the gap"
                );
            } else {
                assert_eq!(spaceline.time, 0, "first spaceline should be 0");
            }
            previous_time = Some(spaceline.time);
            previous_y_stride = Some(y_stride);
        }
        for (spaceline, weight) in &spacelines.buffer0 {
            assert!(
                spaceline.time == previous_time.unwrap() + previous_y_stride.unwrap().as_u64(),
                "mind the gap"
            );
            assert!(
                *weight <= previous_y_stride.unwrap(),
                "should be <= previous_y_stride"
            );
            previous_time = Some(spaceline.time);
            previous_y_stride = Some(*weight);
        }

        if let Some(early_stop) = early_stop {
            // cmk when this is in SpaceByTime, can use self.pixel_policy
            if binning {
                assert!(
                    previous_time.unwrap() + previous_y_stride.unwrap().as_u64() == early_stop,
                    "mind the gap with early_stop"
                );
            } else {
                assert!(
                    previous_time.unwrap() < early_stop
                        && early_stop
                            <= previous_time.unwrap() + previous_y_stride.unwrap().as_u64(),
                    "mind the gap with early_stop"
                );
            }
        }
    }
}
