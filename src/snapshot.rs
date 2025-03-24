use crate::{Error, SpaceByTime, SpaceByTimeMachine};

pub struct Snapshot {
    pub(crate) frame_indexes: Vec<usize>, // TODO make private
    tape_negative_len: usize,
    tape_nonnegative_len: usize,
    pub(crate) space_by_time: SpaceByTime, // TODO make private
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
            space_by_time: space_by_time_machine.space_by_time.clone(),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_png(
        &mut self,
        x_goal: u32,
        y_goal: u32,
        zero_color: [u8; 3],
        one_color: [u8; 3],
    ) -> Result<Vec<u8>, Error> {
        self.space_by_time.to_png(
            self.tape_negative_len,
            self.tape_nonnegative_len,
            x_goal as usize,
            y_goal as usize,
            zero_color,
            one_color,
        )
    }

    pub(crate) fn prepend(mut self, before: SpaceByTime) -> Self {
        self.space_by_time = before.append(self.space_by_time);
        self
    }
}
