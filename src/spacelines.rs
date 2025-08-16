use itertools::Itertools;

use crate::{PixelPolicy, Tape, is_even, power_of_two::PowerOfTwo, spaceline::Spaceline};

#[derive(Clone)]
pub struct Spacelines {
    pub(crate) main: Vec<Spaceline>,                  // TODO make private
    pub(crate) buffer0: Vec<(Spaceline, PowerOfTwo)>, // TODO buffer0 could just be buffer
}

// define a debug that lists the lines of main (one spaceline per line) and then for buffer0, lists the weight and the line, one pair per line
impl core::fmt::Debug for Spacelines {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(formatter, "Spacelines {{")?;
        writeln!(formatter, "  main: [")?;
        for line in &self.main {
            writeln!(formatter, "    {line:?},")?;
        }
        writeln!(formatter, "  ],")?;
        writeln!(formatter, "  buffer0: [")?;
        for (line, weight) in &self.buffer0 {
            writeln!(formatter, "    weight {weight:?}:  {line:?},")?;
        }
        writeln!(formatter, "  ]")?;
        write!(formatter, "}}")
    }
}

impl Spacelines {
    pub(crate) fn new0(select: u8, pixel_policy: PixelPolicy) -> Self {
        Self {
            main: vec![Spaceline::new0(select, pixel_policy)],
            buffer0: Vec::new(),
        }
    }

    pub(crate) fn new_skipped(
        select: u8,
        tape: &Tape,
        x_goal: u32,
        step_index: u64,
        pixel_policy: PixelPolicy,
    ) -> Self {
        let spaceline = Spaceline::new(select, tape, x_goal, step_index, pixel_policy);
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

    #[allow(clippy::min_ident_chars)]
    #[inline]
    pub(crate) fn compress_y_average(&mut self) {
        assert!(self.buffer0.is_empty(), "real assert b2");
        assert!(is_even(self.main.len()), "real assert 11");

        self.main = self
            .main
            .drain(..)
            .tuples()
            .map(|(mut a, b)| {
                assert!(a.tape_start() >= b.tape_start());
                a.merge_simd(&b);
                a
            })
            .collect();
    }

    #[inline]
    pub(crate) fn compress_y_take_first(&mut self, new_stride: PowerOfTwo) {
        assert!(self.buffer0.is_empty(), "real assert e2");
        assert!(is_even(self.main.len()), "real assert e11");
        self.main
            .retain(|spaceline| new_stride.divides_u64(spaceline.time));
    }

    pub(crate) fn last(&self, y_stride: PowerOfTwo, pixel_policy: PixelPolicy) -> Spaceline {
        if self.buffer0.is_empty() {
            // TODO would be nice to remove this clone
            return self.main.last().unwrap().clone();
        }

        match pixel_policy {
            PixelPolicy::Sampling => {
                let (spaceline, _weight) = self.buffer0.first().unwrap();
                assert!(y_stride.divides_u64(spaceline.time));
                spaceline.clone()
            }
            PixelPolicy::Binning => {
                // TODO in the special case in which the sample is 1 and the buffer is 1, can't we just return the buffer's item (as a ref???)
                let mut buffer0 = self.buffer0.clone();

                //We clone because we compress in place.
                loop {
                    let (mut spaceline_last, weight_last) = buffer0.pop().unwrap(); // can't fail
                    // If we have just one item and it has the same weight as main, we're done.
                    if buffer0.is_empty() && weight_last == y_stride {
                        return spaceline_last;
                    }
                    assert!(weight_last < y_stride, "real assert");
                    // Otherwise, we half it's color and double the weight

                    spaceline_last.merge_with_white_simd();

                    Self::push_internal(
                        &mut buffer0,
                        spaceline_last,
                        weight_last.double(),
                        pixel_policy,
                    );
                }
            }
        }
    }

    // TODO make private
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
                    last_spaceline.merge_simd(&spaceline);
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
