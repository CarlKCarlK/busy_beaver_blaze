use core::num::NonZeroU8;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PngDataLayers(pub HashMap<NonZeroU8, Vec<u8>>);

impl PngDataLayers {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }

    #[inline]
    #[must_use]
    pub fn into_inner(self) -> HashMap<NonZeroU8, Vec<u8>> {
        self.0
    }

    #[inline]
    pub fn insert(&mut self, key: NonZeroU8, value: Vec<u8>) -> Option<Vec<u8>> {
        self.0.insert(key, value)
    }
}

impl core::ops::Index<NonZeroU8> for PngDataLayers {
    type Output = Vec<u8>;

    #[inline]
    fn index(&self, index: NonZeroU8) -> &Self::Output {
        &self.0[&index]
    }
}
