extern crate alloc;
use core::num::NonZeroU8;

use aligned_vec::AVec;
use instant::Instant;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::{Machine, PixelPolicy, space_by_time::SpaceByTime, space_time_layers::SpaceTimeLayers};

#[wasm_bindgen]
pub struct SpaceByTimeMachine {
    pub(crate) machine: Machine,
    pub(crate) space_time_layers: SpaceTimeLayers,
}

// impl iterator for spacetime machine
#[allow(clippy::missing_trait_methods)]
impl Iterator for SpaceByTimeMachine {
    type Item = ();

    #[cfg_attr(not(target_arch = "wasm32"), inline)]
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
    /// Creates a `SpaceByTimeMachine` by parsing the program, optionally skipping initial steps, and initializing visualization layers for the given goals and pixel policy.
    ///
    /// # Errors
    /// Returns an error if the program cannot be parsed into a `Machine`, or if the machine halts during the initial skip phase.
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
        // Historically this code used an unordered collection; switching to a BTreeMap
        // made iteration ordered by key. We generally expect the smallest-key layer
        // to correspond to `select == 1`. Add a debug check to capture that
        // invariant during development (does nothing in release builds).
        debug_assert!(
            space_time_layers.first().x_goal
                == SpaceByTime::new0(
                    NonZeroU8::new(1).unwrap(),
                    goal_x,
                    goal_y,
                    if binning {
                        PixelPolicy::Binning
                    } else {
                        PixelPolicy::Sampling
                    }
                )
                .x_goal,
            "Expected first SpaceTimeLayers entry to correspond to select==1"
        );
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

    /// Runs at full speed and records sound visits in roughly two-millisecond slices.
    ///
    /// Returns elapsed microseconds followed by triples of (slice end time in
    /// microseconds, low 32 voice bits, high voice bits). Voices identify the state
    /// performing a transition, or its read symbol. Includes halting transitions.
    /// The ordinary sound-free stepping path is unchanged.
    ///
    /// # Panics
    /// Panics unless `seconds` is finite and in (0, 1], or elapsed time exceeds
    /// the microsecond range of `u32`.
    #[wasm_bindgen]
    pub fn step_for_secs_with_sound(
        &mut self,
        seconds: f32,
        early_stop: Option<u64>,
        read_symbols: bool,
    ) -> Vec<u32> {
        assert!(seconds.is_finite() && seconds > 0.0 && seconds <= 1.0);
        let start = Instant::now();
        let step_count = self.step_count();
        let mut remaining = if let Some(early_stop) = early_stop {
            if step_count >= early_stop {
                return vec![0];
            }
            early_stop - step_count
        } else {
            u64::MAX
        };
        let mut slices = vec![0];
        let mut visits = 0u64;
        let mut last_slice_time = 0;
        'stepping: loop {
            let batch_size = remaining.min(10_000);
            if batch_size == 0 || self.machine.is_halted() {
                break;
            }
            for _step_index in 0..batch_size {
                let voice = if read_symbols {
                    u8::from(self.machine.tape.read(self.machine.tape_index()))
                } else {
                    self.machine.state()
                };
                assert!(voice < 36);
                visits |= 1u64 << voice;
                if self.next().is_none() {
                    break 'stepping;
                }
            }
            assert!(remaining >= batch_size);
            remaining -= batch_size;
            let elapsed = start.elapsed();
            let elapsed_micros =
                u32::try_from(elapsed.as_micros()).expect("sound time must fit in u32");
            assert!(elapsed_micros >= last_slice_time);
            if elapsed_micros - last_slice_time >= 2_000 {
                slices.extend_from_slice(&[elapsed_micros, visits as u32, (visits >> 32) as u32]);
                visits = 0;
                last_slice_time = elapsed_micros;
            }
            if elapsed.as_secs_f32() >= seconds {
                break;
            }
        }
        let elapsed_micros =
            u32::try_from(start.elapsed().as_micros()).expect("sound time must fit in u32");
        if visits != 0 {
            slices.extend_from_slice(&[elapsed_micros, visits as u32, (visits >> 32) as u32]);
        }
        slices[0] = elapsed_micros;
        slices
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
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
    /// Encodes the current space-time visualization into a PNG byte vector using the provided RGB palette.
    ///
    /// The `colors` slice must have a length that is a multiple of 3 (RGB triplets); image dimensions are taken from the first layer.
    ///
    /// # Errors
    /// Returns an error if the `colors` length is not a multiple of 3 or if PNG packing fails in the underlying layers.
    pub fn to_png(&mut self, colors: &[u8]) -> Result<Vec<u8>, String> {
        let (colors, remainder) = colors.as_chunks::<3>();
        if !remainder.is_empty() {
            return Err(format!(
                "Colors length must be a multiple of 3, got {}",
                colors.len() * 3 + remainder.len()
            ));
        }
        // x/y goals pulled from the first layer
        let (x_goal, y_goal) = {
            let space_by_time = self.space_time_layers.first();
            (space_by_time.x_goal, space_by_time.y_goal)
        };

        // 3) Call through and propagate errors instead of unwrap
        let (png_data, _width, _height, _packed_data_list) = self
            .space_time_layers
            .png_data_and_packed_data(
                colors,
                self.machine.tape.negative.len(),
                self.machine.tape.nonnegative.len(),
                (x_goal as usize, y_goal as usize),
            )
            .map_err(|e| e.to_string())?;

        Ok(png_data)
    }

    // TODO why is it step_count and count_ones. That doesn't make sense.
    #[wasm_bindgen]
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn step_count(&self) -> u64 {
        self.step_index() + 1
    }

    #[wasm_bindgen]
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
    #[must_use]
    pub fn count_nonblanks(&self) -> u32 {
        self.machine.count_nonblanks()
    }

    #[wasm_bindgen]
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
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
    #[cfg_attr(not(target_arch = "wasm32"), inline)]
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

    #[cfg_attr(not(target_arch = "wasm32"), inline)]
    #[must_use]
    pub const fn machine(&self) -> &Machine {
        &self.machine
    }

    #[cfg_attr(not(target_arch = "wasm32"), inline)]
    #[must_use]
    pub fn step_index(&self) -> u64 {
        self.space_time_layers.step_index()
    }

    #[cfg_attr(not(target_arch = "wasm32"), inline)]
    #[must_use]
    pub fn state(&self) -> u8 {
        self.machine.state()
    }

    #[cfg_attr(not(target_arch = "wasm32"), inline)]
    #[must_use]
    pub fn tape_index(&self) -> i64 {
        self.machine.tape_index()
    }
}

#[cfg(test)]
mod sound_tests {
    use super::SpaceByTimeMachine;

    #[test]
    fn sound_includes_the_halting_state_and_preserves_rendering() {
        let program = "1RB1RH_1RC1RH_1RH1RH";
        let mut audible = SpaceByTimeMachine::from_str(program, 64, 64, true, 0).unwrap();
        let mut silent = SpaceByTimeMachine::from_str(program, 64, 64, true, 0).unwrap();
        let packet = audible.step_for_secs_with_sound(0.01, None, false);
        let visited = packet[1..]
            .chunks_exact(3)
            .fold(0, |mask, slice| mask | slice[1]);
        assert_eq!(visited, 0b111);
        assert!(!silent.nth_js(100));
        assert!(audible.is_halted());
        assert_eq!(audible.step_count(), silent.step_count());
        assert_eq!(audible.count_nonblanks(), silent.count_nonblanks());
        assert_eq!(audible.to_png(&[]).unwrap(), silent.to_png(&[]).unwrap());
        assert_eq!(audible.step_for_secs_with_sound(0.01, None, false).len(), 1);
    }

    #[test]
    fn sound_reads_extended_symbols_and_respects_step_limits() {
        let program = format!("{}_{}", "zRB".repeat(36), "zLA".repeat(36));
        let mut machine = SpaceByTimeMachine::from_str(&program, 64, 64, false, 0).unwrap();
        assert_eq!(machine.step_for_secs_with_sound(0.01, Some(1), true), [0]);
        let packet = machine.step_for_secs_with_sound(0.01, Some(128), true);
        let high_voices = packet[1..]
            .chunks_exact(3)
            .fold(0, |mask, slice| mask | slice[2]);
        assert_eq!(high_voices, 1 << 3); // Symbol 35 lives in bit 3 of the upper word.
        assert_eq!(machine.step_count(), 128);
        assert!(!machine.is_halted());
        assert_eq!(machine.step_for_secs_with_sound(0.01, Some(128), true), [0]);
    }
}
