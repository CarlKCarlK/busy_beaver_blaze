extern crate alloc;
use alloc::collections::BTreeMap;
use core::ops::Range;
use std::{collections::HashMap, fs};

use aligned_vec::AVec;
use crossbeam::channel::{self, Receiver, Sender};
use instant::Instant;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use wasm_bindgen::prelude::*;

use crate::{Machine, PixelPolicy, Snapshot, find_y_stride, space_by_time::SpaceByTime};

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
    pub fn png_data_and_packed_data(&mut self) -> (Vec<u8>, u32, u32, AVec<u8>) {
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
        part_count_goal: usize,
        program_string: &str,
        goal_x: u32,
        goal_y: u32,
        binning: bool,
        frame_step_indexes: &[u64],
    ) -> Self {
        let (sender, receiver) = channel::unbounded::<(usize, Vec<Snapshot>, Self)>();
        // cmk should part_count be a usize?
        let part_count = Self::run_parts(
            early_stop,
            part_count_goal,
            program_string,
            goal_x,
            goal_y,
            binning,
            frame_step_indexes,
            sender,
        );

        Self::combine_results(goal_x, goal_y, part_count, &receiver)
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

    fn create_range_list(early_stop: u64, part_count_goal: usize, goal_y: u32) -> Vec<Range<u64>> {
        let mut rows_per_part = early_stop.div_ceil(part_count_goal as u64);

        let y_stride = find_y_stride(rows_per_part, goal_y);
        rows_per_part += y_stride.offset_to_align(rows_per_part as usize) as u64;
        assert!(y_stride.divides_u64(rows_per_part), "even?");

        (0..early_stop)
            .step_by(rows_per_part as usize)
            .map(|start| start..(start + rows_per_part).min(early_stop))
            .collect()
    }

    fn build_step_index_to_frame_index(
        frame_step_indexes: &[u64],
        start: u64,
        end: u64,
    ) -> HashMap<u64, Vec<usize>> {
        let mut step_index_to_frame_index: HashMap<u64, Vec<usize>> = HashMap::new();
        for (index, &step_index) in frame_step_indexes.iter().enumerate() {
            if step_index >= start && step_index < end {
                step_index_to_frame_index
                    .entry(step_index)
                    .or_default()
                    .push(index);
            }
        }
        step_index_to_frame_index
    }

    fn generate_snapshots(
        &mut self,
        frame_step_indexes: &[u64],
        start: u64,
        end: u64,
    ) -> Vec<Snapshot> {
        let mut snapshots: Vec<Snapshot> = vec![];
        let step_index_to_frame_index =
            Self::build_step_index_to_frame_index(frame_step_indexes, start, end);
        for step_index in start + 1..end {
            if let Some(frame_indexes) = step_index_to_frame_index.get(&step_index) {
                snapshots.push(Snapshot::new(frame_indexes.clone(), self));
            }
            if self.next().is_none() {
                break;
            }
        }
        snapshots
    }

    // cmk000 need to handle the case of early halting.
    fn run_parts(
        early_stop: u64,
        part_count_goal: usize,
        program_string: &str,
        goal_x: u32,
        goal_y: u32,
        binning: bool,
        frame_step_indexes: &[u64],
        sender: Sender<(usize, Vec<Snapshot>, Self)>,
    ) -> usize {
        assert!(early_stop > 0); // panic if early_stop is 0
        assert!(part_count_goal > 0); // panic if part_count_goal is 0

        let range_list = Self::create_range_list(early_stop, part_count_goal, goal_y);
        let part_count = range_list.len();

        range_list
            .into_par_iter()
            .enumerate()
            .for_each(move |(part_index, range)| {
                let (start, end) = (range.start, range.end);

                let mut space_by_time_machine =
                    Self::from_str(program_string, goal_x, goal_y, binning, start)
                        .expect("Failed to create machine");

                let snapshots =
                    space_by_time_machine.generate_snapshots(frame_step_indexes, start, end);

                let space_by_time = &mut space_by_time_machine.space_by_time;
                let inside_index = space_by_time
                    .y_stride
                    .rem_into_u64(space_by_time.step_index() + 1);
                // This should be 0 on all but the last part
                assert!(inside_index == 0 || part_index == part_count as usize - 1);
                if inside_index == 0 {
                    // We're starting a new set of spacelines, so flush the buffer and compress (if needed)
                    space_by_time.flush_buffer0_and_compress();
                }
                sender
                    .send((part_index, snapshots, space_by_time_machine))
                    .unwrap();
            });
        part_count
    }

    // cmk is it sometimes x_goal and sometimes goal_x????
    fn combine_results(
        x_goal: u32,
        y_goal: u32,
        part_count: usize,
        receiver: &Receiver<(usize, Vec<Snapshot>, Self)>,
    ) -> Self {
        assert!(part_count > 0);
        let mut buffer = BTreeMap::new();

        let mut space_by_time_first_outer = None;
        let mut machine_first = None;
        for next_index in 0..part_count {
            println!("Waiting for: {next_index}");
            // if buffer doesn't start with next_index, then collect something from the channel,
            // and insert it into the buffer and loop again
            while buffer
                .first_key_value()
                .is_none_or(|(&key, _)| key != next_index)
            {
                let (index_received, snapshots, space_by_time_machine) =
                    receiver.recv().expect("Channel closed unexpectedly");
                println!("Received : {index_received}");
                buffer.insert(index_received, (snapshots, space_by_time_machine));
            }
            // pop
            let (popped_index, (snapshots, space_by_time_machine)) =
                buffer.pop_first().expect("Expected next result in buffer");
            assert_eq!(popped_index, next_index);
            println!("Processing: {next_index}");

            // Special processing for the first part
            if next_index == 0 {
                let space_by_time_first = space_by_time_machine.space_by_time;
                assert!(space_by_time_first.spacelines.buffer0.is_empty() || part_count == 1,);

                for mut snapshot_first in snapshots {
                    let png_data = snapshot_first.to_png(x_goal, y_goal).unwrap(); //cmk0 need to handle
                    for frame_index in &snapshot_first.frame_indexes {
                        let cmk_file = format!(r"M:\deldir\bb\frames_test\cmk{frame_index:07}.png");
                        fs::write(cmk_file, &png_data).unwrap();
                    }
                }
                space_by_time_first_outer = Some(space_by_time_first);
                machine_first = Some(space_by_time_machine.machine);
                continue;
            }

            // Process the rest of the parts
            let space_by_time_first = space_by_time_first_outer.unwrap();
            let space_by_time_other = space_by_time_machine.space_by_time;
            Self::assert_empty_buffer_if_not_last_part(
                &space_by_time_other,
                next_index,
                part_count,
            );

            for mut snapshot_other in snapshots {
                // cmk this is convoluted way to combine these two
                snapshot_other = snapshot_other.prepend(space_by_time_first.clone());
                let png_data = snapshot_other.to_png(x_goal, y_goal).unwrap(); //cmk0 need to handle
                for frame_index in &snapshot_other.frame_indexes {
                    let cmk_file = format!(r"M:\deldir\bb\frames_test\cmk{frame_index:07}.png");
                    fs::write(cmk_file, &png_data).unwrap();
                }
            }

            space_by_time_first_outer = Some(space_by_time_first.append(space_by_time_other));
            machine_first = Some(space_by_time_machine.machine);
        }

        Self {
            machine: machine_first.unwrap(),
            space_by_time: space_by_time_first_outer.unwrap(),
        }
    }

    #[inline]
    fn assert_empty_buffer_if_not_last_part(
        space_by_time_other: &SpaceByTime,
        index: usize,
        part_count: usize,
    ) {
        if index < part_count - 1 {
            assert!(
                space_by_time_other.spacelines.buffer0.is_empty(),
                "real assert 2"
            );
        }
    }
}
