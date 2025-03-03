use crate::ALIGN;
use crate::BoolU8;
use aligned_vec::AVec;
use core::ops::RangeInclusive;

#[derive(Debug)]
pub struct Tape {
    pub(crate) nonnegative: AVec<BoolU8>,
    pub(crate) negative: AVec<BoolU8>,
}

impl Default for Tape {
    fn default() -> Self {
        Self {
            nonnegative: AVec::new(ALIGN),
            negative: AVec::new(ALIGN),
        }
    }
}

impl Tape {
    #[inline]
    #[must_use]
    pub fn read(&self, index: i64) -> BoolU8 {
        if index >= 0 {
            self.nonnegative
                .get(index as usize)
                .copied()
                .unwrap_or(BoolU8::FALSE)
        } else {
            self.negative
                .get((-index - 1) as usize)
                .copied()
                .unwrap_or(BoolU8::FALSE)
        }
    }

    #[inline]
    #[allow(clippy::shadow_reuse)]
    pub fn write(&mut self, index: i64, value: BoolU8) {
        let (index, vec) = if index >= 0 {
            (index as usize, &mut self.nonnegative)
        } else {
            ((-index - 1) as usize, &mut self.negative) // cmk this code appear more than once
        };

        if index == vec.len() {
            // We are exactly one index beyond the current length
            vec.push(value);
        } else {
            // Assert that we're never more than one index beyond
            debug_assert!(
                index < vec.len(),
                "Index is more than one beyond current length!"
            );
            vec[index] = value;
        }
    }

    pub fn count_ones(&self) -> usize {
        self.nonnegative
            .iter()
            .chain(self.negative.iter()) // Combine both vectors
            .map(usize::from)
            .sum()
    }

    #[cfg(test)]
    #[allow(clippy::min_ident_chars)]
    #[must_use]
    pub fn index_range_to_string(&self, range: RangeInclusive<i64>) -> String {
        let mut s = String::new();
        for i in range {
            s.push_str(&self.read(i).to_string());
        }
        s
    }

    #[inline]
    #[must_use]
    pub fn min_index(&self) -> i64 {
        -(self.negative.len() as i64)
    }

    #[inline]
    #[must_use]
    pub fn max_index(&self) -> i64 {
        self.nonnegative.len() as i64 - 1
    }

    #[inline]
    #[allow(dead_code)]
    #[must_use]
    pub fn width(&self) -> u64 {
        (self.max_index() - self.min_index() + 1) as u64
    }
}
