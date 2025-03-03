use crate::pixel::Pixel;
use crate::tape::Tape;
use crate::{PixelPolicy, PowerOfTwo};
use aligned_vec::AVec;
use smallvec::SmallVec;

#[derive(Clone, Debug)]
pub struct Spaceline {
    pub sample: PowerOfTwo,
    pub negative: AVec<Pixel>,
    pub nonnegative: AVec<Pixel>,
    pub time: u64,
    pub pixel_policy: PixelPolicy,
}

impl Spaceline {
    pub fn new0(pixel_policy: PixelPolicy) -> Self {
        let mut vector = AVec::new(1);
        vector.push(Pixel::WHITE);
        Self {
            sample: PowerOfTwo::ONE,
            negative: AVec::new(64),
            nonnegative: vector,
            time: 0,
            pixel_policy,
        }
    }

    #[inline]
    pub fn pixel_index(&self, index: usize) -> Pixel {
        let negative_len = self.negative.len();
        if index < negative_len {
            self.negative[negative_len - 1 - index]
        } else {
            self.nonnegative[index - negative_len]
        }
    }

    #[inline]
    pub fn pixel_index_unbounded(&self, index: usize) -> Pixel {
        let negative_len = self.negative.len();
        if index < negative_len {
            self.negative
                .get(negative_len - 1 - index)
                .copied()
                .unwrap_or_default()
        } else {
            self.nonnegative
                .get(index - negative_len)
                .copied()
                .unwrap_or_default()
        }
    }

    #[inline]
    pub fn new2(
        sample: PowerOfTwo,
        start: i64,
        pixels: AVec<Pixel>,
        time: u64,
        pixel_policy: PixelPolicy,
    ) -> Self {
        let mut result = Self {
            sample,
            negative: AVec::new(64),
            nonnegative: pixels,
            time,
            pixel_policy,
        };
        while result.tape_start() > start {
            result.negative.insert(0, result.nonnegative.remove(0));
        }
        result
    }

    #[inline]
    pub fn pixel_index_mut(&mut self, index: usize) -> &mut Pixel {
        let negative_len = self.negative.len();
        if index < negative_len {
            &mut self.negative[negative_len - 1 - index]
        } else {
            &mut self.nonnegative[index - negative_len]
        }
    }

    #[inline]
    pub fn tape_start(&self) -> i64 {
        -((self.sample * self.negative.len()) as i64)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nonnegative.len() + self.negative.len()
    }

    #[inline]
    pub fn resample_if_needed(&mut self, sample: PowerOfTwo) {
        // cmk0000 seems overly messy
        assert!(!self.nonnegative.is_empty(), "real assert a");
        assert!(
            self.sample.divides_i64(self.tape_start()),
            "Start must be a multiple of the sample rate"
        );
        if sample == self.sample {
            return;
        }
        let cells_to_add = sample.rem_euclid_into(self.tape_start());
        let new_tape_start = self.tape_start() - cells_to_add;
        let old_items_to_add = self.sample.divide_into(cells_to_add);
        let old_items_per_new = sample / self.sample;
        let old_items_per_new_u64 = old_items_per_new.as_u64();
        let old_items_per_new_usize = old_items_per_new_u64 as usize;
        assert!(sample >= self.sample, "real assert 12");
        let old_items_to_use = old_items_per_new.as_u64() - old_items_to_add as u64;
        assert!(old_items_to_use <= self.len() as u64, "real assert d10");
        // let down_step = sample.saturating_div(self.pixel_policy);
        let pixel0 = Pixel::merge_slice_down_sample(
            &self.pixel_range(0, old_items_to_use as usize),
            old_items_to_add as usize,
            self.pixel_policy,
        );
        let mut new_index = 0usize;
        *self.pixel_index_mut(new_index) = pixel0;
        new_index += 1;
        let value_len = self.len() as u64;
        // let down_size_usize = down_step.as_usize();
        for old_index in (old_items_to_use..value_len).step_by(old_items_per_new_usize) {
            let old_end = (old_index + old_items_to_use).min(value_len);
            let slice = &self.pixel_range(old_index as usize, old_end as usize);
            let old_items_to_add_inner = old_items_per_new_u64 - (old_end - old_index);
            *self.pixel_index_mut(new_index) = Pixel::merge_slice_down_sample(
                slice,
                old_items_to_add_inner as usize,
                self.pixel_policy,
            );
            new_index += 1;
        }
        self.pixel_restart(new_tape_start, new_index, sample);
    }

    pub fn pixel_range(&self, start: usize, end: usize) -> Vec<Pixel> {
        assert!(
            start <= end,
            "start index {start} must be <= end index {end}"
        );
        let mut result = Vec::with_capacity(end - start);
        for i in start..end {
            result.push(self.pixel_index(i));
        }
        result
    }

    #[inline]
    pub fn pixel_restart(&mut self, tape_start: i64, len: usize, sample: PowerOfTwo) {
        self.sample = sample;
        assert!(self.tape_start() <= tape_start, "real assert 11");
        while self.tape_start() < tape_start {
            self.nonnegative.insert(0, self.negative.remove(0));
        }
        assert!(self.len() >= len, "real assert 12");
        while self.len() > len {
            self.nonnegative.pop();
        }
        assert!(self.len() == len, "real assert 13");
    }

    #[inline]
    pub fn merge(&mut self, other: &Self) {
        assert!(self.time < other.time, "real assert 2");
        assert!(self.sample <= other.sample, "real assert 3");
        assert!(self.tape_start() >= other.tape_start(), "real assert 4");
        self.resample_if_needed(other.sample);
        assert!(self.sample == other.sample, "real assert 5b");
        assert!(self.tape_start() >= other.tape_start(), "real assert 6c");
        while self.tape_start() > other.tape_start() {
            self.negative.push(Pixel::WHITE);
        }
        assert!(self.tape_start() == other.tape_start(), "real assert 6c");
        while self.len() < other.len() {
            self.nonnegative.push(Pixel::WHITE);
        }
        assert!(self.len() == other.len(), "real assert 6d");
        Pixel::slice_merge(&mut self.nonnegative, &other.nonnegative);
    }

    #[inline]
    pub fn new(tape: &Tape, x_goal: u32, step_index: u64, pixel_policy: PixelPolicy) -> Self {
        let tape_min_index = tape.min_index();
        let tape_max_index = tape.max_index();
        let tape_width = (tape_max_index - tape_min_index + 1) as u64;
        let x_sample = crate::sample_rate(tape_width, x_goal);
        if step_index % 10_000_000 == 0 {
            println!(
                "cmk Spaceline::new step_index {}, tape width {:?} ({}..={}), x_sample {:?}, x_goal {:?}",
                step_index,
                tape_width,
                tape_min_index,
                tape_max_index,
                x_sample.as_usize(),
                x_goal
            );
        }
        match pixel_policy {
            PixelPolicy::Binning => {
                let (negative, nonnegative) = match x_sample {
                    PowerOfTwo::ONE | PowerOfTwo::TWO | PowerOfTwo::FOUR => (
                        crate::average_with_iterators(&tape.negative, x_sample),
                        crate::average_with_iterators(&tape.nonnegative, x_sample),
                    ),
                    PowerOfTwo::EIGHT => (
                        crate::average_with_simd::<8>(&tape.negative, x_sample),
                        crate::average_with_simd::<8>(&tape.nonnegative, x_sample),
                    ),
                    PowerOfTwo::SIXTEEN => (
                        crate::average_with_simd::<16>(&tape.negative, x_sample),
                        crate::average_with_simd::<16>(&tape.nonnegative, x_sample),
                    ),
                    PowerOfTwo::THIRTY_TWO => (
                        crate::average_with_simd::<32>(&tape.negative, x_sample),
                        crate::average_with_simd::<32>(&tape.nonnegative, x_sample),
                    ),
                    _ => (
                        crate::average_with_simd::<64>(&tape.negative, x_sample),
                        crate::average_with_simd::<64>(&tape.nonnegative, x_sample),
                    ),
                };
                Self {
                    sample: x_sample,
                    negative,
                    nonnegative,
                    time: step_index,
                    pixel_policy,
                }
            }
            PixelPolicy::Sampling => {
                let sample_start: i64 = tape_min_index - x_sample.rem_euclid_into(tape_min_index);
                assert!(
                    sample_start <= tape_min_index
                        && x_sample.divides_i64(sample_start)
                        && tape_min_index - sample_start < x_sample.as_u64() as i64,
                    "real assert b1"
                );
                let mut pixels = AVec::with_capacity(64, x_goal as usize * 2);
                // cmk00 does this need to be a special case?
                if x_sample == PowerOfTwo::ONE {
                    for sample_index in sample_start..=tape_max_index {
                        pixels.push(tape.read(sample_index).into());
                    }
                } else {
                    for sample_index in (sample_start..=tape_max_index).step_by(x_sample.as_usize())
                    {
                        pixels.push(tape.read(sample_index).into());
                    }
                }
                Self::new2(x_sample, sample_start, pixels, step_index, pixel_policy)
            }
        }
    }

    #[inline]
    pub fn redo_pixel(
        &mut self,
        previous_tape_index: i64,
        tape: &Tape,
        x_goal: u32,
        step_index: u64,
        pixel_policy: PixelPolicy,
    ) -> bool {
        self.time = step_index;
        let tape_min_index = tape.min_index();
        let tape_max_index = tape.max_index();
        assert!(tape_min_index <= previous_tape_index && previous_tape_index <= tape_max_index);
        let tape_width = (tape_max_index - tape_min_index + 1) as u64;
        let x_sample = crate::sample_rate(tape_width, x_goal);
        let x_sample_usize = x_sample.as_usize();
        if self.sample != x_sample {
            return false;
        }
        let (part, pixels, part_index) = if previous_tape_index < 0 {
            let part_index = (-previous_tape_index - 1) as u64;
            (&tape.negative, &mut self.negative, part_index)
        } else {
            let part_index = previous_tape_index as u64;
            (&tape.nonnegative, &mut self.nonnegative, part_index)
        };
        let pixel_index = x_sample.divide_into(part_index) as usize;
        assert!(pixel_index <= pixels.len());
        if pixel_index == pixels.len() {
            pixels.push(Pixel::WHITE);
        }
        let pixel = &mut pixels[pixel_index];
        let tape_slice_start = x_sample * pixel_index;
        match pixel_policy {
            PixelPolicy::Binning => {
                // cmk000000 could be faster??
                let sum: u32 = (tape_slice_start..tape_slice_start + x_sample_usize)
                    .filter_map(|i| part.get(i).map(u32::from))
                    .sum();
                let mean = x_sample.divide_into(sum * 255) as u8;
                *pixel = Pixel(mean);
            }
            PixelPolicy::Sampling => {
                *pixel = Pixel::from(part[tape_slice_start]);
            }
        }
        true
    }
}

// ...existing code continues...
