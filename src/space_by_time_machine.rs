use core::ops::Range;
use std::{collections::HashMap, fs};

use instant::Instant;
use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use wasm_bindgen::prelude::*;

use crate::{
    Machine, PixelPolicy, PowerOfTwo, Snapshot, is_even, sample_rate, space_by_time::SpaceByTime,
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

        let mut space_by_time_machine =
            Self::combine_results(goal_x, goal_y, snapshots_and_space_by_time_machines);

        loop {
            space_by_time_machine.compress_cmk3_y_if_needed(goal_y, early_stop, binning);
            space_by_time_machine.reduce_buffer0();

            if space_by_time_machine.space_by_time.spacelines.len() <= goal_y as usize * 2 {
                break;
            }
        }
        space_by_time_machine
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
        let y_stride = space_by_time_first.y_stride;

        for mut snapshot_first in snapshots_first {
            let png_data = snapshot_first.to_png(x_goal, y_goal).unwrap(); //cmk0000 need to handle
            for frame_index in &snapshot_first.frame_indexes {
                let cmk_file = format!(r"M:\deldir\bb\frames_test\cmk{frame_index:07}.png");
                fs::write(cmk_file, &png_data).unwrap();
            }
        }

        let mut index: usize = 0;
        for (_snapshots_other, space_by_time_machine_other) in results_iter {
            index += 1;
            let space_by_time = space_by_time_machine_other.space_by_time;
            let spacelines = space_by_time.spacelines;
            let main = spacelines.main;
            let buffer0 = spacelines.buffer0;
            if index < part_count as usize - 1 {
                assert!(buffer0.is_empty(), "real assert 2");
            }

            let (mut previous_y_stride, mut previous_time) = if space_by_time.y_stride == y_stride {
                for spaceline in main {
                    space_by_time_first.spacelines.main.push(spaceline);
                }
                (
                    space_by_time_first.y_stride,
                    space_by_time_first.spacelines.main.last().unwrap().time,
                )
            } else {
                assert!(
                    index == part_count as usize - 1,
                    "y_stride can only be less on the last part"
                );
                for spaceline in main {
                    space_by_time_first
                        .spacelines
                        .buffer0
                        .push((spaceline, space_by_time.y_stride));
                }
                (
                    space_by_time.y_stride,
                    space_by_time_first
                        .spacelines
                        .buffer0
                        .last()
                        .unwrap()
                        .0
                        .time,
                )
            };

            for (spaceline, weight) in buffer0 {
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

            space_by_time_machine_first.machine = space_by_time_machine_other.machine;
        }

        space_by_time_machine_first
    }

    // cmk0 understand the `compress_cmk*` functions
    fn compress_cmk3_y_if_needed(&mut self, goal_y: u32, early_stop: u64, binning: bool) {
        let space_by_time = &mut self.space_by_time;
        Self::audit_one(space_by_time, None, None, early_stop, binning);

        loop {
            Self::audit_one(space_by_time, None, None, early_stop, binning);
            let len = space_by_time.spacelines.len();
            assert!(len > 0, "real assert 5");
            if len < goal_y as usize * 2 {
                break;
            }
            if !is_even(space_by_time.spacelines.main.len()) {
                Self::audit_one(space_by_time, None, None, early_stop, binning);
                // println!("is odd: Spacelines {:?}", space_by_time.spacelines);
                let last = space_by_time.spacelines.main.pop().unwrap();
                space_by_time
                    .spacelines
                    .buffer0
                    .insert(0, (last, space_by_time.y_stride));
                Self::audit_one(space_by_time, None, None, early_stop, binning);
            }
            // println!("Spacelines: {:?}", space_by_time.spacelines);

            assert!(
                is_even(space_by_time.spacelines.main.len()),
                "real assert 6"
            );

            Self::audit_one(space_by_time, None, None, early_stop, binning);
            space_by_time.spacelines.main = space_by_time
                .spacelines
                .main
                .drain(..)
                .tuples()
                .map(|(mut first, second)| {
                    // println!("tuple a: {:?} b: {:?}", first.time, second.time);
                    assert!(first.tape_start() >= second.tape_start(), "real assert 4a");

                    if binning {
                        // cmk00 remove from loop?
                        first.merge(&second);
                    } else {
                        /* do nothing */
                    }
                    first
                })
                .collect();
            space_by_time.y_stride = space_by_time.y_stride.double();
            // println!("After binning or sampling: {:?}", space_by_time.spacelines);
            Self::audit_one(space_by_time, None, None, early_stop, binning);
        }
    }

    // fn double_audit(self, early_stop: u64, binning: bool) -> Self {
    //     let single_results = [self];
    //     Self::audit_results(&single_results, 1, early_stop, binning);
    //     let result = single_results.into_iter().next().unwrap();
    //     let space_by_time = &result.space_by_time;
    //     Self::audit_one(space_by_time, None, None, early_stop, binning);
    //     result
    // }

    // fn audit_results(results: &[Self], part_count: u64, early_stop: u64, binning: bool) {
    //     assert_eq!(results.len() as u64, part_count);
    //     let mut previous_y_stride = None;
    //     let mut previous_time = None;
    //     for (part_index, space_by_time_machine) in results.iter().enumerate() {
    //         let space_by_time = &space_by_time_machine.space_by_time;
    //         let y_stride = space_by_time.y_stride;
    //         if part_index > 0 && part_index < part_count as usize - 1 {
    //             assert_eq!(
    //                 y_stride,
    //                 previous_y_stride.unwrap(),
    //                 "from part to part, the stride should be the same"
    //             );
    //         }
    //         previous_y_stride = Some(y_stride);
    //         let spacelines = &space_by_time.spacelines;
    //         let main = &spacelines.main;
    //         for spaceline in main {
    //             if let Some(previous_time) = previous_time {
    //                 assert!(
    //                     spaceline.time == previous_time + y_stride.as_u64(),
    //                     "mind the gap"
    //                 );
    //             }
    //             previous_time = Some(spaceline.time);
    //         }
    //         if part_index < part_count as usize - 1 {
    //             assert!(
    //                 spacelines.buffer0.is_empty(),
    //                 "only the last part can have a buffer"
    //             );
    //         } else {
    //             for (spaceline, weight) in &spacelines.buffer0 {
    //                 assert!(
    //                     spaceline.time
    //                         == previous_time.unwrap() + previous_y_stride.unwrap().as_u64(),
    //                     "mind the gap"
    //                 );
    //                 assert!(
    //                     *weight <= previous_y_stride.unwrap(),
    //                     "should be <= previous_y_stride"
    //                 );
    //                 previous_time = Some(spaceline.time);
    //                 previous_y_stride = Some(*weight);
    //             }
    //         }
    //     }
    //     if binning {
    //         assert!(
    //             previous_time.unwrap() + previous_y_stride.unwrap().as_u64() == early_stop,
    //             "mind the gap with early_stop"
    //         );
    //     } else {
    //         assert!(
    //             previous_time.unwrap() < early_stop
    //                 && early_stop <= previous_time.unwrap() + previous_y_stride.unwrap().as_u64(),
    //             "mind the gap with early_stop"
    //         );
    //     }
    // }
    // cmk move this to the SpaceByTime struct

    #[inline]
    #[allow(unused_variables)]
    fn audit_one(
        space_by_time: &SpaceByTime,
        previous_y_stride: Option<PowerOfTwo>,
        previous_time: Option<u64>,
        stop: u64,
        binning: bool,
    ) {
        // on debug compiles call audit_one_internal otherwise do nothing
        #[cfg(debug_assertions)]
        Self::audit_one_internal(
            space_by_time,
            previous_y_stride,
            previous_time,
            stop,
            binning,
        );
    }

    #[cfg(debug_assertions)]
    fn audit_one_internal(
        space_by_time: &SpaceByTime,
        mut previous_y_stride: Option<PowerOfTwo>,
        mut previous_time: Option<u64>,
        stop: u64,
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
        if binning {
            // cmk why called stop here, but early_stop elsewhere
            assert!(
                previous_time.unwrap() + previous_y_stride.unwrap().as_u64() == stop,
                "mind the gap with early_stop"
            );
        } else {
            assert!(
                previous_time.unwrap() < stop
                    && stop <= previous_time.unwrap() + previous_y_stride.unwrap().as_u64(),
                "mind the gap with early_stop"
            );
        }
    }

    fn reduce_buffer0(&mut self) {
        let space_by_time = &mut self.space_by_time;

        // println!(
        //     "main's time: {} y_stride {:?}",
        //     &space_by_time.spacelines.main.last().unwrap().time,
        //     &space_by_time.y_stride
        // );
        // for (spaceline, weight) in &space_by_time.spacelines.buffer0 {
        //     println!("time: {}, weight: {weight:?}", spaceline.time);
        // }

        // take the buffer0 leaving it empty
        let buffer_old = core::mem::take(&mut space_by_time.spacelines.buffer0);
        let buffer0 = &mut space_by_time.spacelines.buffer0;
        let mut old_weight = None;
        for (spaceline, weight) in buffer_old {
            assert!(
                old_weight.is_none() || old_weight.unwrap() >= weight,
                "should be monotonic"
            );
            assert!(weight <= space_by_time.y_stride, "should be <= y_stride");
            if weight == space_by_time.y_stride {
                // This is a special case where we have a spaceline that is exactly the y_stride, so we can just push it to the main buffer
                space_by_time.spacelines.main.push(spaceline);
                continue;
            }
            Spacelines::push_internal(buffer0, spaceline, weight, space_by_time.pixel_policy);
            old_weight = Some(weight);

            // println!("=== +{weight:?}");
            // for (spaceline_x, weight_x) in buffer0.iter() {
            //     println!("after time: {}, weight: {weight_x:?}", spaceline_x.time);
            // }

            if buffer0.len() == 1 && buffer0.first().unwrap().1 == space_by_time.y_stride {
                // This is a special case where we have a spaceline that is exactly the y_stride, so we can just push it to the main buffer
                let (spaceline_z, _weight_z) = buffer0.pop().unwrap();
                space_by_time.spacelines.main.push(spaceline_z);
            }
        }
        // println!(
        //     "again main's time {}, y_stride {:?}",
        //     &space_by_time.spacelines.main.last().unwrap().time,
        //     &space_by_time.y_stride
        // );
        // for (spaceline, weight) in &space_by_time.spacelines.buffer0 {
        //     println!("again: time: {}, weight: {weight:?}", spaceline.time);
        // }
        // println!("{:?}", space_by_time.spacelines);
    }
}
