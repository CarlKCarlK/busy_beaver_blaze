extern crate alloc;
use alloc::vec::Vec;

use core::ops::Range;
use crossbeam::channel::{self, Receiver, Sender};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::{
    Snapshot, SpaceByTimeMachine, find_y_stride, message0::Message0,
    space_time_layers::SpaceTimeLayers,
};
use alloc::collections::BinaryHeap;
use std::{
    collections::HashMap,
    thread::{self, JoinHandle},
};

pub struct PngDataIterator {
    receiver1: Receiver<(usize, u64, Vec<u8>)>, // frame_index, step_index, png_data
    // Optionally hold the join handles so that threads are joined when the iterator is dropped.
    run_handle: Option<JoinHandle<()>>,
    combine_handle: Option<JoinHandle<SpaceByTimeMachine>>,
}
impl PngDataIterator {
    /// Creates a new `PngDataIterator` by spawning the necessary worker threads.
    // TODO: Review all the "allow"'s
    #[must_use]
    #[allow(clippy::missing_panics_doc, clippy::too_many_arguments)]
    pub fn new(
        early_stop: u64,
        part_count_goal: usize,
        program_string: &str,
        colors: &[[u8; 3]],
        goal_x: u32,
        goal_y: u32,
        binning: bool,
        frame_index_to_step_index: &[u64],
    ) -> Self {
        assert!(part_count_goal > 0, "part_count_goal must be > 0");
        assert!(early_stop > 0); // panic if early_stop is 0

        // cmk000 for multithreading reasons???? this must be a Vec
        let colors_owned: Vec<[u8; 3]> = colors.to_vec();

        // TODO for now require that values in frame_index_to_step_index be increasing and in range. This could be removed later.
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
        let (sender0, receiver0) = channel::unbounded::<Message0>();
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
            Self::combine_results(
                colors_owned.as_slice(),
                goal_x,
                goal_y,
                part_count,
                &receiver0,
                &sender1,
            )
        });

        Self {
            receiver1,
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
        sender0: Sender<Message0>,
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

            Self::generate_snapshots(&mut space_by_time_machine,
                frame_index_to_step_index.as_slice(),
                step_start,
                step_end,
                &sender0,
                part_index,
            );

            println!("Part {part_index}/{part_count}, have snapshotted desired steps from {step_start} to {step_end}.");


            for space_by_time in space_by_time_machine.space_time_layers.values_mut() {
                // let space_by_time = &mut space_by_time_machine.space_by_time;
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
            }

            let message = Message0::SpaceByTimeMachine { part_index , space_by_time_machine };
            sender0.send(message).unwrap();

            println!("Part {part_index}/{part_count} has transmitted all snapshots and the space_by_time_machine");

        });
    }

    // TODO is it sometimes x_goal and sometimes goal_x????
    #[allow(clippy::too_many_lines)]
    fn combine_results(
        colors: &[[u8; 3]],
        x_goal: u32,
        y_goal: u32,

        part_count: usize,
        receiver0: &Receiver<Message0>,
        sender1: &Sender<(usize, u64, Vec<u8>)>, // frame_index, step_index, png_data
    ) -> SpaceByTimeMachine {
        assert!(part_count > 0);
        let mut buffer: BinaryHeap<Message0> = BinaryHeap::new();

        let mut space_time_layers_outer: Option<SpaceTimeLayers> = None;
        let mut machine_first = None;
        for next_part_index in 0..part_count {
            println!("Waiting for: {next_part_index}");
            // if buffer doesn't start with next_index, then collect something from the channel,
            // and insert it into the buffer and loop again
            while buffer
                .peek()
                .is_none_or(|msg| msg.part_index() != next_part_index)
            {
                let message0 = receiver0.recv().expect("Channel closed unexpectedly");
                println!(
                    "Received: part_index={}, step_index={}",
                    message0.part_index(),
                    message0.step_index()
                );
                buffer.push(message0);
            }

            'SAME_PART: loop {
                let message0 = buffer.pop().expect("Expected next result in buffer");
                assert_eq!(message0.part_index(), next_part_index);
                match message0 {
                    Message0::Snapshot {
                        part_index,
                        mut snapshot,
                    } => {
                        assert_eq!(part_index, next_part_index);
                        println!(
                            "Processing snapshot: part_index={next_part_index}, frame_indexes={:?}",
                            snapshot.frame_indexes
                        );

                        if next_part_index == 0 {
                            let step_index = snapshot.step_index();
                            let png_data =
                                snapshot.to_png(colors.as_ref(), x_goal, y_goal).unwrap(); // TODO need to handle
                            let (last, all_but_last) = snapshot.frame_indexes.split_last().unwrap();
                            // The same step can be rendered to multiple places
                            for index in all_but_last {
                                sender1
                                    .send((*index, step_index, png_data.clone()))
                                    .expect("Failed to send PNG data");
                            }
                            sender1
                                .send((*last, step_index, png_data))
                                .expect("Failed to send PNG data");
                        } else {
                            let mut space_time_layers_first =
                                space_time_layers_outer.as_ref().unwrap().clone();
                            space_time_layers_first.merge(snapshot.space_time_layers);
                            snapshot.space_time_layers = space_time_layers_first;
                            let png_data =
                                snapshot.to_png(colors.as_ref(), x_goal, y_goal).unwrap(); // TODO need to handle
                            let (last, all_but_last) = snapshot.frame_indexes.split_last().unwrap();
                            let step_index = snapshot.step_index();
                            for index in all_but_last {
                                sender1
                                    .send((*index, step_index, png_data.clone()))
                                    .expect("Failed to send PNG data");
                            }
                            sender1
                                .send((*last, step_index, png_data))
                                .expect("Failed to send PNG data");
                        }

                        // Keep reading until we get another with the same part_index. There must be one
                        // because the SpaceByTimeMachine message is last.
                        while buffer
                            .peek()
                            .is_none_or(|msg| msg.part_index() != next_part_index)
                        {
                            let message00 = receiver0.recv().expect("Channel closed unexpectedly");
                            println!(
                                "Received: part_index={}, step_index={}",
                                message00.part_index(),
                                message00.step_index()
                            );
                            buffer.push(message00);
                        }
                    }
                    Message0::SpaceByTimeMachine {
                        part_index,
                        space_by_time_machine,
                    } => {
                        assert_eq!(part_index, next_part_index);

                        println!("Processing space_by_time_machine: part_index={next_part_index}");

                        if next_part_index == 0 {
                            assert!(
                                part_count == 1
                                    || space_by_time_machine.space_time_layers.values().all(
                                        |space_by_time0| space_by_time0
                                            .spacelines
                                            .buffer0
                                            .is_empty()
                                    ),
                                "Expected all layer buffers empty for non-last parts"
                            );
                            assert!(space_time_layers_outer.is_none());
                            space_time_layers_outer = Some(space_by_time_machine.space_time_layers);
                        } else {
                            let mut space_time_layers_first = space_time_layers_outer.unwrap();
                            let space_time_layers_other = space_by_time_machine.space_time_layers;
                            Self::assert_empty_buffer_if_not_last_part(
                                &space_time_layers_other,
                                next_part_index,
                                part_count,
                            );

                            space_time_layers_first.merge(space_time_layers_other);
                            space_time_layers_outer = Some(space_time_layers_first);
                        }
                        // TODO ??? we don't need the whole space_by_time_machine, just the machine
                        machine_first = Some(space_by_time_machine.machine);

                        // ...
                        break 'SAME_PART;
                    }
                }
            }
        }

        SpaceByTimeMachine {
            machine: machine_first.unwrap(),
            space_time_layers: space_time_layers_outer.unwrap(),
        }
    }

    #[inline]
    fn assert_empty_buffer_if_not_last_part(
        space_time_layer_other: &SpaceTimeLayers,
        index: usize,
        part_count: usize,
    ) {
        if index < part_count - 1 {
            assert!(
                space_time_layer_other.first().spacelines.buffer0.is_empty(),
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
    pub fn into_space_by_time_machine(mut self) -> SpaceByTimeMachine {
        let Some(combine_handle) = self.combine_handle.take() else {
            panic!("combine_handle missing");
        };
        let space_by_time_machine = combine_handle.join().unwrap();
        let Some(run_handle) = self.run_handle.take() else {
            panic!("run_handle missing");
        };
        let _ = run_handle.join();
        // TODO??? need to do more to shutdown threads?

        space_by_time_machine
    }

    fn generate_snapshots(
        space_by_time_machine: &mut SpaceByTimeMachine,
        frame_index_to_step_index: &[u64],
        step_start: u64,
        step_end: u64,
        sender0: &Sender<Message0>,
        part_index: usize,
    ) {
        // println!("---\n{step_start}..{step_end}: {frame_index_to_step_index:?}");
        let step_index_to_frame_index =
            Self::build_step_index_to_frame_index(frame_index_to_step_index, step_start, step_end);
        // println!("---\n{step_start}..{step_end}: {step_index_to_frame_index:?}");
        for step_index in step_start..step_end - 1 {
            if let Some(frame_indexes) = step_index_to_frame_index.get(&step_index) {
                let snapshot = Snapshot::new(frame_indexes.clone(), space_by_time_machine);
                let message = Message0::Snapshot {
                    part_index,
                    snapshot,
                };
                sender0.send(message).unwrap();
            }
            if space_by_time_machine.next().is_none() {
                break;
            }
        }
        // TODO not sure if should show last frame if self.next was none
        if let Some(frame_indexes) = step_index_to_frame_index.get(&(step_end - 1)) {
            let snapshot = Snapshot::new(frame_indexes.clone(), space_by_time_machine);
            let message = Message0::Snapshot {
                part_index,
                snapshot,
            };
            sender0.send(message).unwrap();
        }
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
    // cmk0000 make a struct
    type Item = (u64, Vec<u8>); // step_index, png_data

    fn next(&mut self) -> Option<Self::Item> {
        // This will block until a new frame is available or the channel is closed.
        self.receiver1
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
