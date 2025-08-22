use std::{collections::HashMap, num::NonZeroU8};

use crate::{Error, SpaceByTimeMachine, space_time_layers::SpaceTimeLayers};

pub struct Snapshot {
    pub(crate) frame_indexes: Vec<usize>, // TODO make private
    tape_negative_len: usize,
    tape_nonnegative_len: usize,
    pub(crate) space_time_layers: SpaceTimeLayers, // TODO make private
}

// given a frame_index and SpaceTimeMachine, create a snapshot
impl Snapshot {
    pub(crate) fn new(
        frame_indexes: Vec<usize>,
        space_by_time_machine: &SpaceByTimeMachine,
    ) -> Self {
        Self {
            frame_indexes,
            tape_negative_len: space_by_time_machine.machine().tape.negative.len(),
            tape_nonnegative_len: space_by_time_machine.machine().tape.nonnegative.len(),
            space_time_layers: space_by_time_machine.space_time_layers.clone(),
        }
    }

    pub(crate) fn step_index(&self) -> u64 {
        self.space_time_layers.step_index() // cmk unwrap
    }

    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_png(
        &mut self,
        x_goal: u32,
        y_goal: u32,
    ) -> Result<HashMap<NonZeroU8, Vec<u8>>, Error> {
        let mut select_to_png: HashMap<NonZeroU8, Vec<u8>> = HashMap::new();
        for (select, space_by_time) in &mut self.space_time_layers {
            let png = space_by_time.to_png(
                self.tape_negative_len,
                self.tape_nonnegative_len,
                x_goal as usize,
                y_goal as usize,
            )?;
            select_to_png.insert(*select, png);
        }
        Ok(select_to_png)
    }

    // pub(crate) fn prepend(mut self, before: SpaceByTime) -> Self {
    //     self.space_by_time = before.append(self.space_by_time);
    //     self
    // }
}
