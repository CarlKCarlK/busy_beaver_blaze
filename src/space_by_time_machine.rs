extern crate alloc;
use core::num::NonZeroU8;

use aligned_vec::AVec;
use instant::Instant;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::{
    DEFAULT_COLORS, Machine, PixelPolicy, space_by_time::SpaceByTime,
    space_time_layers::SpaceTimeLayers,
};

#[wasm_bindgen]
pub struct SpaceByTimeMachine {
    pub(crate) machine: Machine,
    pub(crate) space_time_layers: SpaceTimeLayers,
}

// impl iterator for spacetime machine
#[allow(clippy::missing_trait_methods)]
impl Iterator for SpaceByTimeMachine {
    type Item = ();

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let previous_tape_index = self.machine.next()?;
        for space_by_time in self.space_time_layers.values_mut() {
            space_by_time.snapshot(&self.machine, previous_tape_index);
        }
        Some(())
    }
}

#[wasm_bindgen]
#[allow(clippy::min_ident_chars)]
#[allow(clippy::missing_panics_doc)]
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
        let symbol_count = machine.program.symbol_count;
        let mut space_time_layers = SpaceTimeLayers::default();
        for select in 1u8..symbol_count {
            let select = NonZeroU8::new(select).unwrap();
            let space_by_time = SpaceByTime::new_skipped(
                select,
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
            space_time_layers.insert(select, space_by_time);
        }
        Ok(Self {
            machine,
            space_time_layers,
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
    pub fn to_png(&mut self, colors: &[u8]) -> Result<Vec<u8>, String> {
        let colors = if colors.is_empty() {
            DEFAULT_COLORS
        } else {
            let (colors, remainder) = colors.as_chunks::<3>();
            if !remainder.is_empty() {
                return Err(format!(
                    "Colors length must be a multiple of 3, got {}",
                    colors.len() * 3 + remainder.len()
                ));
            }
            colors
        };
        if colors.len() < 2 {
            return Err(format!(
                "Colors length must be at least 2, got {}",
                colors.len()
            ));
        }
        let colors: Vec<_> = core::iter::once(colors[0])
            .chain(colors[1..].iter().copied().cycle())
            .take(self.machine.program.symbol_count as usize)
            .collect();
        // x/y goals pulled from the first layer
        let (x_goal, y_goal) = {
            let space_by_time = self.space_time_layers.first();
            (space_by_time.x_goal, space_by_time.y_goal)
        };

        // 3) Call through and propagate errors instead of unwrap
        let (png_data, _width, _height, _packed_data_list) = self
            .space_time_layers
            .png_data_and_packed_data(
                &colors,
                self.machine.tape.negative.len(),
                self.machine.tape.nonnegative.len(),
                (x_goal as usize, y_goal as usize),
            )
            .map_err(|e| e.to_string())?;

        Ok(png_data)
    }

    // TODO why is it step_count and count_ones. That doesn't make sense.
    #[wasm_bindgen]
    #[inline]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn step_count(&self) -> u64 {
        self.step_index() + 1
    }

    #[wasm_bindgen]
    #[inline]
    #[must_use]
    pub fn count_nonblanks(&self) -> u32 {
        self.machine.count_nonblanks()
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
    pub fn png_data_and_packed_data(
        &mut self,
        colors: &[[u8; 3]],
    ) -> (Vec<u8>, u32, u32, Vec<AVec<u8>>) {
        let space_time_layers = &mut self.space_time_layers;
        let (goal_width, goal_height) = {
            let space_by_time = space_time_layers.first();
            (space_by_time.x_goal as usize, space_by_time.y_goal as usize)
        };
        space_time_layers
            .png_data_and_packed_data(
                colors,
                self.machine.tape.negative.len(),
                self.machine.tape.nonnegative.len(),
                (goal_width, goal_height),
            )
            .expect("Failed to collect packed data")
    }

    #[inline]
    #[must_use]
    pub const fn machine(&self) -> &Machine {
        &self.machine
    }

    #[inline]
    #[must_use]
    pub fn step_index(&self) -> u64 {
        self.space_time_layers.step_index()
    } // cmk make const?

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
