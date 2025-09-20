use crate::ALIGN;
use crate::Symbol;
use aligned_vec::AVec;

#[derive(Debug)]
pub struct Tape {
    pub(crate) negative: AVec<Symbol>,
    pub(crate) nonnegative: AVec<Symbol>,
}

impl Default for Tape {
    fn default() -> Self {
        let mut nonnegative = AVec::new(ALIGN);
        nonnegative.push(Symbol::STATE_ZERO);
        Self {
            negative: AVec::new(ALIGN),
            nonnegative,
        }
    }
}

impl Tape {
    #[inline]
    #[must_use]
    /// Returns the symbol at `index`, or `STATE_ZERO` if `index` is beyond the current tape bounds.
    ///
    /// # Panics
    /// Panics if the tape index magnitude cannot be represented as a `usize` on this target.
    pub fn read(&self, index: i64) -> Symbol {
        if index >= 0 {
            let i = usize::try_from(index).expect("tape index must fit in usize");
            self.nonnegative
                .get(i)
                .copied()
                .unwrap_or(Symbol::STATE_ZERO)
        } else {
            // Avoid overflow on i64::MIN by computing magnitude as -(index + 1)
            let mag = -(index + 1);
            let i = usize::try_from(mag).expect("negative tape index magnitude must fit in usize");
            self.negative.get(i).copied().unwrap_or(Symbol::STATE_ZERO)
        }
    }

    #[inline]
    #[allow(clippy::shadow_unrelated)]
    /// # Panics
    /// Panics if the tape index magnitude cannot be represented as a `usize` on this target.
    pub fn write(&mut self, index: i64, value: Symbol) {
        let (index, vec) = if index >= 0 {
            let index = usize::try_from(index).expect("tape index must fit in usize");
            (index, &mut self.nonnegative)
        } else {
            // Avoid overflow on i64::MIN by computing magnitude as -(index + 1)
            let mag = -(index + 1);
            let index =
                usize::try_from(mag).expect("negative tape index magnitude must fit in usize");
            (index, &mut self.negative) // TODO this code appear more than once
        };

        if index == vec.len() {
            // We are exactly one index beyond the current length
            vec.push(value);
        } else {
            // assert that we're never more than one index beyond
            debug_assert!(
                index < vec.len(),
                "Index is more than one beyond current length!"
            );
            vec[index] = value;
        }
    }

    #[must_use]
    pub fn count_nonblanks(&self) -> usize {
        self.nonnegative
            .iter()
            .chain(self.negative.iter())
            .map(|&x| usize::from(x != Symbol::STATE_ZERO))
            .sum()
    }

    #[cfg(test)]
    #[must_use]
    pub fn index_range_to_string(&self, range: core::ops::RangeInclusive<i64>) -> String {
        let mut result_string = String::new();
        for i in range {
            result_string.push_str(&self.read(i).to_string());
        }
        result_string
    }
}
