use aligned_vec::AVec;
use itertools::Itertools;

use crate::{
    ALIGN, PixelPolicy, Tape, is_even, pixel::Pixel, power_of_two::PowerOfTwo, spaceline::Spaceline,
};

pub(crate) struct Spacelines {
    pub(crate) main: Vec<Spaceline>,                  // cmk make private
    pub(crate) buffer0: Vec<(Spaceline, PowerOfTwo)>, // cmk0 better names
}

impl Spacelines {
    pub(crate) fn new0(pixel_policy: PixelPolicy) -> Self {
        Self {
            main: vec![Spaceline::new0(pixel_policy)],
            buffer0: Vec::new(),
        }
    }

    pub(crate) fn new_skipped(
        tape: &Tape,
        x_goal: u32,
        step_index: u64,
        pixel_policy: PixelPolicy,
    ) -> Self {
        let spaceline = Spaceline::new(tape, x_goal, step_index, pixel_policy);
        Self {
            main: vec![spaceline],
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
        assert!(is_even(self.main.len()), "real assert 11");
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
    pub(crate) fn compress_take_first(&mut self, new_stride: PowerOfTwo) {
        assert!(self.buffer0.is_empty(), "real assert e2");
        assert!(is_even(self.main.len()), "real assert e11");
        // println!("cmk compress_take_first");
        self.main
            .retain(|spaceline| new_stride.divides_u64(spaceline.time));
    }

    pub(crate) fn last(
        &self,
        step_index: u64,
        y_stride: PowerOfTwo,
        pixel_policy: PixelPolicy,
    ) -> Spaceline {
        if self.buffer0.is_empty() {
            // cmk would be nice to remove this clone
            return self.main.last().unwrap().clone();
        }
        // cmk in the special case in which the sample is 1 and the buffer is 1, can't we just return the buffer's item (as a ref???)

        let buffer_last = self.buffer0.last().unwrap();
        let spaceline_last = &buffer_last.0;
        let time = spaceline_last.time;
        let start = spaceline_last.tape_start();
        let x_stride = spaceline_last.x_stride;
        println!("cmk last x_stride: {x_stride:?}");
        let last_inside_index = y_stride.rem_into_u64(step_index);

        // cmk we have to clone because we compress in place (clone only half???)
        let mut buffer0 = self.buffer0.clone();
        match pixel_policy {
            PixelPolicy::Sampling => {
                // what's the small number I need to add to last_inside_index to create a multiple of y_stride?
                let smallest = y_stride.offset_to_align(last_inside_index as usize);
                if smallest != 0 {
                    let mut empty_pixels =
                        AVec::<Pixel>::with_capacity(ALIGN, spaceline_last.len());
                    empty_pixels.resize(spaceline_last.len(), Pixel::WHITE);

                    let empty = Spaceline::new2(
                        x_stride,
                        start,
                        empty_pixels,
                        time + smallest as u64,
                        spaceline_last.pixel_policy,
                    );
                    Self::push_internal(&mut buffer0, empty, PowerOfTwo::ONE);
                }
            }
            PixelPolicy::Binning => {
                for inside_index in last_inside_index + 1..y_stride.as_u64() {
                    let mut empty_pixels =
                        AVec::<Pixel>::with_capacity(ALIGN, spaceline_last.len());
                    empty_pixels.resize(spaceline_last.len(), Pixel::WHITE);
                    let empty = Spaceline::new2(
                        x_stride,
                        start,
                        empty_pixels,
                        time + inside_index - last_inside_index,
                        spaceline_last.pixel_policy,
                    );
                    Self::push_internal(&mut buffer0, empty, PowerOfTwo::ONE);
                }
            }
        }

        assert!(buffer0.len() == 1, "real assert b3");
        buffer0.pop().unwrap().0
    }

    // cmk000 make private
    #[inline]
    pub(crate) fn push_internal(
        buffer0: &mut Vec<(Spaceline, PowerOfTwo)>,
        mut spaceline: Spaceline,
        mut weight: PowerOfTwo,
    ) {
        while let Some((_last_mute_spaceline, last_mut_weight)) = buffer0.last_mut() {
            // If current weight is smaller, just append to buffer
            if weight < *last_mut_weight {
                buffer0.push((spaceline, weight));
                return;
            }

            // cmk change back to debug_assert (or not)
            assert!(
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
    pub(crate) fn push(&mut self, spaceline: Spaceline) {
        Self::push_internal(&mut self.buffer0, spaceline, PowerOfTwo::ONE);
    }
}
