use aligned_vec::AVec;
use itertools::Itertools;
use num_traits::Pow;

use crate::{
    ALIGN, PixelPolicy, fast_is_even, pixel::Pixel, power_of_two::PowerOfTwo, spaceline::Spaceline,
};

pub(crate) struct Spacelines {
    main: Vec<Spaceline>,
    buffer0: Vec<(Spaceline, PowerOfTwo)>, // cmk0 better names
}

impl Spacelines {
    pub(crate) fn new(pixel_policy: PixelPolicy) -> Self {
        Self {
            main: vec![Spaceline::new0(pixel_policy)],
            buffer0: Vec::new(),
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.main.len() + usize::from(!self.buffer0.is_empty())
    }

    pub(crate) fn get<'a>(&'a self, index: usize, last: &'a Spaceline) -> &'a Spaceline {
        if index == self.len() - 1 {
            last
        } else {
            &self.main[index]
        }
    }

    pub(crate) fn flush_buffer0(&mut self) {
        // We now have a buffer that needs to be flushed at the end
        if !self.buffer0.is_empty() {
            assert!(self.buffer0.len() == 1, "real assert 13");
            self.main.push(self.buffer0.pop().unwrap().0);
        }
    }

    #[allow(clippy::min_ident_chars)]
    #[inline]
    pub(crate) fn compress_average(&mut self) {
        assert!(self.buffer0.is_empty(), "real assert b2");
        assert!(fast_is_even(self.main.len()), "real assert 11");
        // println!("cmk compress_average");

        self.main = self
            .main
            .drain(..)
            .tuples()
            .map(|(mut a, b)| {
                assert!(a.tape_start() >= b.tape_start(), "real assert 4a");
                a.merge(&b);
                a
            })
            .collect();
    }

    #[inline]
    pub(crate) fn compress_take_first(&mut self, new_sample: PowerOfTwo) {
        assert!(self.buffer0.is_empty(), "real assert e2");
        assert!(fast_is_even(self.main.len()), "real assert e11");
        // println!("cmk compress_take_first");
        self.main
            .retain(|spaceline| new_sample.divides_u64(spaceline.time));
    }

    pub(crate) fn last(
        &mut self,
        step_index: u64,
        y_sample: PowerOfTwo,
        pixel_policy: PixelPolicy,
    ) -> Spaceline {
        let y_smoothness = match pixel_policy {
            PixelPolicy::Sampling => PowerOfTwo::ONE,
            PixelPolicy::Binning => PowerOfTwo::MAX,
        };

        if self.buffer0.is_empty() {
            // cmk would be nice to remove this clone
            return self.main.last().unwrap().clone();
        }
        // cmk in the special case in which the sample is 1 and the buffer is 1, can't we just return the buffer's item (as a ref???)

        let buffer_last = self.buffer0.last().unwrap();
        let spaceline_last = &buffer_last.0;
        let weight = buffer_last.1; // cmk0000 should this be used?
        let time = spaceline_last.time;
        let start = spaceline_last.tape_start();
        let x_sample = spaceline_last.sample;
        let last_inside_index = y_sample.rem_into_u64(step_index);

        // cmk we have to clone because we compress in place (clone only half???)
        let mut buffer0 = self.buffer0.clone();
        for inside_index in last_inside_index + 1..y_sample.as_u64() {
            let down_step = y_sample.saturating_div(y_smoothness);
            if !down_step.divides_u64(inside_index) {
                continue;
            }
            let inside_inside_index = down_step.divide_into(inside_index);

            let empty_pixels = AVec::from_iter(
                ALIGN,
                core::iter::repeat_n(Pixel::WHITE, spaceline_last.len()),
            );
            let empty = Spaceline::new2(
                x_sample,
                start,
                empty_pixels,
                time + inside_index - last_inside_index,
                spaceline_last.pixel_policy,
            );
            Self::push_internal(&mut buffer0, inside_inside_index, empty, PowerOfTwo::ONE);
        }
        assert!(buffer0.len() == 1, "real assert b3");
        buffer0.pop().unwrap().0
    }

    // pub(crate) fn last(
    //     &mut self,
    //     step_index: u64,
    //     y_sample: PowerOfTwo,
    //     pixel_policy: PixelPolicy,
    // ) -> Spaceline {
    //     if self.buffer0.is_empty() {
    //         // cmk would be nice to remove this clone
    //         return self.main.last().unwrap().clone();
    //     }
    //     // cmk in the special case in which the sample is 1 and the buffer is 1, can't we just return the buffer's item (as a ref???)

    //     let buffer_last = self.buffer0.last().unwrap();
    //     let spaceline_last = &buffer_last.0;
    //     let weight = buffer_last.1; // cmk0000 should this be used?
    //     let time = spaceline_last.time;
    //     let start = spaceline_last.tape_start();
    //     let x_sample = spaceline_last.sample;
    //     let last_inside_index = y_sample.rem_into_u64(step_index);

    //     // cmk we have to clone because we compress in place (clone only half???)
    //     let mut buffer0 = self.buffer0.clone();
    //     for inside_index in last_inside_index + 1..y_sample.as_u64() { //cmk000000000 why are we counting one by one when sampling?
    //         let inside_inside_index = match pixel_policy {
    //             PixelPolicy::Sampling => {
    //                 let inside_inside_index = y_sample.divide_into(inside_index);
    //                 if inside_inside_index != 0 {
    //                     continue;
    //                 }
    //                 0
    //             }
    //             PixelPolicy::Binning => inside_index,
    //         };

    //         let mut empty_pixels = AVec::<Pixel>::with_capacity(ALIGN, spaceline_last.len());
    //         empty_pixels.resize(spaceline_last.len(), Pixel::WHITE);
    //         let empty = Spaceline::new2(
    //             x_sample,
    //             start,
    //             empty_pixels,
    //             time + inside_index - last_inside_index,
    //             spaceline_last.pixel_policy,
    //         );
    //         Self::push_internal(&mut buffer0, inside_inside_index, empty, PowerOfTwo::ONE);
    //     }
    //     assert!(buffer0.len() == 1, "real assert b3");
    //     buffer0.pop().unwrap().0
    // }

    #[inline]
    fn push_internal(
        buffer0: &mut Vec<(Spaceline, PowerOfTwo)>,
        _inside_index: u64,
        mut spaceline: Spaceline,
        mut weight: PowerOfTwo,
    ) {
        while let Some((_last_mute_spaceline, last_mut_weight)) = buffer0.last_mut() {
            // If current weight is smaller, just append to buffer
            if weight < *last_mut_weight {
                buffer0.push((spaceline, weight));
                return;
            }

            debug_assert!(
                weight == *last_mut_weight,
                "Weight equality invariant violation"
            );

            // Get ownership of the last element by popping
            let (mut last_spaceline, last_weight) = buffer0.pop().unwrap();

            // Merge spacelines and double weight
            last_spaceline.merge(&spaceline);

            // Continue with the merged spaceline and doubled weight
            spaceline = last_spaceline;
            weight = last_weight.double();

            // If buffer is now empty, push and return
            if buffer0.is_empty() {
                buffer0.push((spaceline, weight));
                return;
            }
        }

        // Handle empty buffer case
        buffer0.push((spaceline, weight));
    }

    #[inline]
    pub(crate) fn push(&mut self, inside_index: u64, spaceline: Spaceline) {
        Self::push_internal(&mut self.buffer0, inside_index, spaceline, PowerOfTwo::ONE);
    }
}
