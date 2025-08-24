use crate::{
    Error, SpaceByTimeMachine, encode_png_colors, png_data_layers::PngDataLayers,
    space_time_layers::SpaceTimeLayers,
};

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
        colors: &[[u8; 3]],
        x_goal: u32,
        y_goal: u32,
    ) -> Result<Vec<u8>, Error> {
        // cmk00000 somewhere else there is a similar x_y look for it option<(u32, u32)>
        let mut packed_data_list = Vec::new();
        let mut x1: Option<u32> = None;
        let mut y1: Option<u32> = None;
        for (_select, space_by_time) in &mut self.space_time_layers {
            let (x, y, packed_data) = space_by_time.to_packed_data(
                self.tape_negative_len,
                self.tape_nonnegative_len,
                x_goal as usize,
                y_goal as usize,
            )?;
            x1 = Some(x);
            y1 = Some(y);
            packed_data_list.push(packed_data);
        }
        let png = encode_png_colors(
            x1.unwrap(),
            y1.unwrap(),
            colors,
            packed_data_list.as_slice(),
        )?;

        Ok(png)
    }

    // pub(crate) fn prepend(mut self, before: SpaceByTime) -> Self {
    //     self.space_by_time = before.append(self.space_by_time);
    //     self
    // }
}
