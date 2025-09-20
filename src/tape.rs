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
    pub fn read(&self, index: i64) -> Symbol {
        if index >= 0 {
            self.nonnegative
                .get(index as usize)
                .copied()
                .unwrap_or(Symbol::STATE_ZERO)
        } else {
            self.negative
                .get((-index - 1) as usize)
                .copied()
                .unwrap_or(Symbol::STATE_ZERO)
        }
    }

    #[inline]
    pub fn write(&mut self, index: i64, value: Symbol) {
        let (index, vec) = if index >= 0 {
            (index as usize, &mut self.nonnegative)
        } else {
            ((-index - 1) as usize, &mut self.negative) // TODO this code appear more than once
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
