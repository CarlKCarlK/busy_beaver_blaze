extern crate alloc;
use alloc::collections::BTreeMap;
use core::num::NonZeroU8;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PngDataLayers(pub BTreeMap<NonZeroU8, Vec<u8>>);

impl PngDataLayers {
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self(BTreeMap::new())
    }

    #[inline]
    #[must_use]
    pub fn into_inner(self) -> BTreeMap<NonZeroU8, Vec<u8>> {
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
