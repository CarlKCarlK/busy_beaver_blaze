extern crate alloc;
use alloc::{collections::BTreeMap, vec::Vec};
use core::mem::ManuallyDrop;
use core::ops::Range;
use crossbeam::channel::{self, Receiver, Sender};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::{Snapshot, SpaceByTime, SpaceByTimeMachine, find_y_stride};
use std::{
    collections::HashMap,
    thread::{self, JoinHandle},
}; // Keep thread imports as they're not in alloc/core

pub struct PngDataIterator {
    receiver: Receiver<(usize, u64, Vec<u8>)>,
    // Optionally hold the join handles so that threads are joined when the iterator is dropped.
    run_handle: Option<JoinHandle<()>>,
    combine_handle: Option<JoinHandle<SpaceByTimeMachine>>,
}
impl PngDataIterator {
    /// Creates a new `PngDataIterator` by spawning the necessary worker threads.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn new(
        early_stop: u64,
        part_count_goal: usize,
        program_string: &str,
        goal_x: u32,
        goal_y: u32,
        binning: bool,
        frame_index_to_step_index: &[u64],
    ) -> Self {
        assert!(part_count_goal > 0, "part_count_goal must be > 0");
        assert!(early_stop > 0); // panic if early_stop is 0

        // cmk for now require that values in frame_index_to_step_index be increasing and in range. This could be removed later.

        assert!(
            frame_index_to_step_index
                .windows(2)
                .all(|window| window[0] <= window[1]),
            "frame_index_to_step_index must be monotonically increasing"
        );
        assert!(
            frame_index_to_step_index
                .iter()
                .all(|value| value < &early_stop),
            "frame_index_to_step_index values must be less than early_stop"
        );

        // Create the range list based on provided parameters.
        let step_index_ranges = Self::create_step_index_ranges(early_stop, part_count_goal, goal_y);
        let part_count = step_index_ranges.len();

        // Set up channels for inter-thread communication.
        let (sender0, receiver0) =
            channel::unbounded::<(usize, Vec<Snapshot>, SpaceByTimeMachine)>();
        let (sender1, receiver1) = channel::unbounded::<(usize, u64, Vec<u8>)>();

        // Clone any data needed by the spawned threads.
        let frame_index_to_step_index_clone = frame_index_to_step_index.to_vec();
        let program_string_clone = program_string.to_owned();

        // Spawn the thread that processes parts.
        let run_handle = thread::spawn(move || {
            Self::run_parts(
                early_stop,
                step_index_ranges,
                program_string_clone,
                goal_x,
                goal_y,
                binning,
                frame_index_to_step_index_clone,
                sender0,
            );
        });

        // Spawn the thread that combines the results into PNG data.
        let combine_handle = thread::spawn(move || {
            Self::combine_results(goal_x, goal_y, part_count, &receiver0, &sender1)
        });

        Self {
            receiver: receiver1,
            run_handle: Some(run_handle),
            combine_handle: Some(combine_handle),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_parts(
        early_stop: u64,
        step_index_ranges: Vec<Range<u64>>,
        program_string: String,
        goal_x: u32,
        goal_y: u32,
        binning: bool,
        frame_index_to_step_index: Vec<u64>,
        sender: Sender<(usize, Vec<Snapshot>, SpaceByTimeMachine)>,
    ) {
        assert!(early_stop > 0); // panic if early_stop is 0

        let part_count = step_index_ranges.len();

        step_index_ranges
        .into_par_iter()
        // .into_iter()
        .enumerate()
        .for_each(move |(part_index, step_index_range)| {
            let (step_start, step_end) = (step_index_range.start, step_index_range.end);
            println!(
                "Part {part_index}/{part_count}, working on visualizing step_index {step_start}..{step_end}."
            );

            let mut space_by_time_machine =
                SpaceByTimeMachine::from_str(&program_string, goal_x, goal_y, binning, step_start)
                    .expect("Failed to create machine");

            println!("Part {part_index}/{part_count}, have fast-forwarded {step_start} steps before visualization.");

            let snapshots = Self::generate_snapshots(&mut space_by_time_machine,
                frame_index_to_step_index.as_slice(),
                step_start,
                step_end,
            );

            println!("Part {part_index}/{part_count}, have snapshotted desired steps from {step_start} to {step_end}.");


            let space_by_time = &mut space_by_time_machine.space_by_time;
            let inside_index = space_by_time
                .y_stride
                .rem_into_u64(space_by_time.vis_step + 1);
            // This should be 0 on all but the last part
            // println!(
            //     "part {part_index}/{part_count} inside_index: {inside_index}, y_stride: {:?}, step_index: {:?}",
            //     space_by_time.y_stride, space_by_time.step_index()
            // );
            assert!(inside_index == 0 || part_index == part_count - 1);
            if inside_index == 0 {
                // We're starting a new set of spacelines, so flush the buffer and compress (if needed)
                space_by_time.flush_buffer0_and_compress();
            }
            sender
                .send((part_index, snapshots, space_by_time_machine))
                .unwrap();

            println!("Part {part_index}/{part_count}, have transmitted snapshots");

        });
    }

    // cmk is it sometimes x_goal and sometimes goal_x????
    fn combine_results(
        x_goal: u32,
        y_goal: u32,
        part_count: usize,
        receiver0: &Receiver<(usize, Vec<Snapshot>, SpaceByTimeMachine)>,
        sender1: &Sender<(usize, u64, Vec<u8>)>,
    ) -> SpaceByTimeMachine {
        assert!(part_count > 0);
        let mut buffer = BTreeMap::new();

        let mut space_by_time_first_outer = None;
        let mut machine_first = None;
        for next_part_index in 0..part_count {
            println!("Waiting for: {next_part_index}");
            // if buffer doesn't start with next_index, then collect something from the channel,
            // and insert it into the buffer and loop again
            while buffer
                .first_key_value()
                .is_none_or(|(&key, _)| key != next_part_index)
            {
                let (part_index_received, snapshots, space_by_time_machine) =
                    receiver0.recv().expect("Channel closed unexpectedly");
                println!("Received : {part_index_received}");
                buffer.insert(part_index_received, (snapshots, space_by_time_machine));
            }
            // pop
            let (popped_index, (snapshots, space_by_time_machine)) =
                buffer.pop_first().expect("Expected next result in buffer");
            assert_eq!(popped_index, next_part_index);
            println!("Processing: {next_part_index}");

            // Special processing for the first part
            if next_part_index == 0 {
                let space_by_time_first = space_by_time_machine.space_by_time;
                assert!(space_by_time_first.spacelines.buffer0.is_empty() || part_count == 1,);

                for mut snapshot_first in snapshots {
                    let png_data = snapshot_first.to_png(x_goal, y_goal).unwrap(); //cmk0 need to handle
                    let (last, all_but_last) = snapshot_first.frame_indexes.split_last().unwrap();
                    let step_index = snapshot_first.space_by_time.step_index();
                    // The same step can be rendered to multiple places
                    for index in all_but_last {
                        sender1
                            .send((*index, step_index, png_data.clone()))
                            .expect("Failed to send PNG data");
                    }
                    sender1
                        .send((*last, step_index, png_data))
                        .expect("Failed to send PNG data");
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
                next_part_index,
                part_count,
            );

            for mut snapshot_other in snapshots {
                snapshot_other = snapshot_other.prepend(space_by_time_first.clone());
                let png_data = snapshot_other.to_png(x_goal, y_goal).unwrap(); //cmk0 need to handle
                let (last, all_but_last) = snapshot_other.frame_indexes.split_last().unwrap();
                let step_index = snapshot_other.space_by_time.step_index();
                for index in all_but_last {
                    sender1
                        .send((*index, step_index, png_data.clone()))
                        .expect("Failed to send PNG data");
                }
                sender1
                    .send((*last, step_index, png_data))
                    .expect("Failed to send PNG data");
            }

            space_by_time_first_outer = Some(space_by_time_first.append(space_by_time_other));
            machine_first = Some(space_by_time_machine.machine);
        }

        SpaceByTimeMachine {
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

    fn create_step_index_ranges(
        early_stop: u64,
        part_count_goal: usize,
        goal_y: u32,
    ) -> Vec<Range<u64>> {
        let mut rows_per_part = early_stop.div_ceil(part_count_goal as u64);

        let y_stride = find_y_stride(rows_per_part, goal_y);
        rows_per_part += y_stride.offset_to_align(rows_per_part as usize) as u64;
        assert!(y_stride.divides_u64(rows_per_part), "even?");

        (0..early_stop)
            .step_by(rows_per_part as usize)
            .map(|start| start..(start + rows_per_part).min(early_stop))
            .collect()
    }

    /// Consumes this iterator and returns the final `SpaceByTimeMachine`
    /// by joining the combine thread.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn into_space_by_time_machine(self) -> SpaceByTimeMachine {
        // Wrap self in ManuallyDrop so we can extract its fields without triggering Drop.
        let this = ManuallyDrop::new(self);
        // SAFETY: We are manually extracting the fields from `this` because we
        // know we're consuming `self` entirely.
        let receiver = unsafe { core::ptr::read(&this.receiver) };
        let run_handle = unsafe { core::ptr::read(&this.run_handle) };
        let combine_handle = unsafe { core::ptr::read(&this.combine_handle) };

        // Drop the receiver to signal no more frames.
        drop(receiver);
        // Join the combine thread to get the final machine.
        let machine = combine_handle
            .expect("combine_handle missing")
            .join()
            .expect("combine thread panicked");
        // Join the run_handle if it still exists.
        if let Some(handle) = run_handle {
            let _ = handle.join();
        }
        machine
    }

    fn generate_snapshots(
        space_by_time_machine: &mut SpaceByTimeMachine,
        frame_index_to_step_index: &[u64],
        step_start: u64,
        step_end: u64,
    ) -> Vec<Snapshot> {
        // println!("---\n{step_start}..{step_end}: {frame_index_to_step_index:?}");
        let mut snapshots: Vec<Snapshot> = vec![];
        let step_index_to_frame_index =
            Self::build_step_index_to_frame_index(frame_index_to_step_index, step_start, step_end);
        // println!("---\n{step_start}..{step_end}: {step_index_to_frame_index:?}");
        for step_index in step_start..step_end - 1 {
            if let Some(frame_indexes) = step_index_to_frame_index.get(&step_index) {
                // println!(
                //     "step_index {step_index} ({}), {:?}",
                //     space_by_time_machine.space_by_time.step_index(),
                //     frame_indexes
                // );
                snapshots.push(Snapshot::new(frame_indexes.clone(), space_by_time_machine));
            }
            if space_by_time_machine.next().is_none() {
                break;
            }
        }
        // cmk00 not sure if should show last frame if self.next was none
        if let Some(frame_indexes) = step_index_to_frame_index.get(&(step_end - 1)) {
            // println!(
            //     "step_index {} ({}), {:?}",
            //     step_end - 1,
            //     space_by_time_machine.space_by_time.step_index(),
            //     frame_indexes
            // );
            snapshots.push(Snapshot::new(frame_indexes.clone(), space_by_time_machine));
        }
        snapshots
    }

    fn build_step_index_to_frame_index(
        frame_index_to_step_index: &[u64],
        start: u64,
        end: u64,
    ) -> HashMap<u64, Vec<usize>> {
        let mut step_index_to_frame_index: HashMap<u64, Vec<usize>> = HashMap::new();
        for (index, &step_index) in frame_index_to_step_index.iter().enumerate() {
            if step_index >= start && step_index < end {
                step_index_to_frame_index
                    .entry(step_index)
                    .or_default()
                    .push(index);
            }
        }
        step_index_to_frame_index
    }
}

#[allow(clippy::missing_trait_methods)]
impl Iterator for PngDataIterator {
    type Item = (u64, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        // This will block until a new frame is available or the channel is closed.
        self.receiver
            .recv()
            .ok()
            .map(|(_frame_index, step_index, png_data)| (step_index, png_data))
    }
}

impl Drop for PngDataIterator {
    fn drop(&mut self) {
        // Ensure that the spawned threads are joined.
        if let Some(handle) = self.run_handle.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.combine_handle.take() {
            let _ = handle.join();
        }
    }
}

// cmk should we be using async instead of threads for the two?
