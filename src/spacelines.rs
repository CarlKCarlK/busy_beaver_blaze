use aligned_vec::AVec;
use itertools::Itertools;

use crate::{
    ALIGN, PixelPolicy, Tape, is_even,
    pixel::{self, Pixel},
    power_of_two::PowerOfTwo,
    spaceline::Spaceline,
};

pub(crate) struct Spacelines {
    pub(crate) main: Vec<Spaceline>,                  // cmk make private
    pub(crate) buffer0: Vec<(Spaceline, PowerOfTwo)>, // cmk0 better names
}

// define a debug that lists the lines of main (one spaceline per line) and then for buffer0, lists the weight and the line, one pair per line
impl core::fmt::Debug for Spacelines {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Spacelines {{")?;
        writeln!(f, "  main: [")?;
        for line in &self.main {
            writeln!(f, "    {line:?},")?;
        }
        writeln!(f, "  ],")?;
        writeln!(f, "  buffer0: [")?;
        for (line, weight) in &self.buffer0 {
            writeln!(f, "    weight {weight:?}:  {line:?},")?;
        }
        writeln!(f, "  ]")?;
        write!(f, "}}")
    }
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

        if pixel_policy == PixelPolicy::Sampling {
            let (spaceline, weight) = self.buffer0.first().unwrap();
            // cmk remove assert!(*weight == y_stride || weight.double() == y_stride);
            assert!(y_stride.divides_u64(spaceline.time));
            let clone = spaceline.clone();
            return clone;
        }

        // cmk in the special case in which the sample is 1 and the buffer is 1, can't we just return the buffer's item (as a ref???)
        let mut buffer0 = self.buffer0.clone();

        // cmk we have to clone because we compress in place (clone only half???)
        loop {
            let (mut spaceline_last, weight_last) = buffer0.pop().unwrap(); // can't fail
            // If we have just one item and it has the same weight as main, we're done.
            if buffer0.is_empty() && weight_last == y_stride {
                return spaceline_last;
            }
            assert!(weight_last < y_stride, "real assert");
            // Otherwise, we half it's color and double the weight
            match pixel_policy {
                PixelPolicy::Sampling => {
                    todo!()
                } // cmk000000 this is wrong. Should use empty line unless the sampling line divides evenginly
                PixelPolicy::Binning => spaceline_last.merge_with_white(),
            }

            Self::push_internal(
                &mut buffer0,
                spaceline_last,
                weight_last.double(),
                pixel_policy,
            );
        }
    }

    // cmk000 make private
    #[inline]
    pub(crate) fn push_internal(
        buffer0: &mut Vec<(Spaceline, PowerOfTwo)>,
        mut spaceline: Spaceline,
        mut weight: PowerOfTwo,
        pixel_policy: PixelPolicy,
    ) {
        // Sampling & Averaging 3
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

            match pixel_policy {
                PixelPolicy::Sampling => {
                    spaceline = last_spaceline;
                }
                PixelPolicy::Binning => {
                    // Merge spacelines and double weight
                    last_spaceline.merge(&spaceline);
                    // Continue with the merged spaceline
                    spaceline = last_spaceline;
                }
            }
            // ... and doubled weight
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
    pub(crate) fn push(
        &mut self,
        spaceline: Spaceline,
        pixel_policy: PixelPolicy,
        weight: PowerOfTwo,
    ) {
        Self::push_internal(&mut self.buffer0, spaceline, weight, pixel_policy);
    }
}
