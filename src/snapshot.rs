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

    #[inline]
    pub(crate) fn step_index(&self) -> u64 {
        self.space_time_layers.step_index()
    }

    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_png(
        &mut self,
        colors: &[[u8; 3]],
        goal_width: u32,
        goal_height: u32,
    ) -> Result<Vec<u8>, Error> {
        let (png_data, _width, _height, _packed_data_list) =
            self.space_time_layers.png_data_and_packed_data(
                colors,
                self.tape_negative_len,
                self.tape_nonnegative_len,
                (goal_width as usize, goal_height as usize),
            )?;
        Ok(png_data)
    }
}
