// cmk00 should this be space_by_time_layers???
use core::num::NonZeroU8;
use std::collections::HashMap;

use crate::SpaceByTime;

/// Layers is an anonymous wrapper around `HashMap`<`NonZeroU8`, `SpaceByTime`>
#[derive(Clone, Default)]
pub struct SpaceTimeLayers(HashMap<NonZeroU8, SpaceByTime>);

impl SpaceTimeLayers {
    #[inline]
    pub fn first(&self) -> &SpaceByTime {
        assert!(!self.0.is_empty()); // cmk00
        &self.0[&NonZeroU8::new(1).unwrap()]
    }

    #[inline]
    pub fn insert(&mut self, key: NonZeroU8, value: SpaceByTime) {
        self.0.insert(key, value);
    }

    #[inline]
    #[must_use]
    pub fn step_index(&self) -> u64 {
        self.first().step_index()
    }

    #[inline]
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut SpaceByTime> {
        self.0.values_mut()
    }

    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &SpaceByTime> {
        self.0.values()
    }

    #[inline]
    pub fn get_mut(&mut self, key: NonZeroU8) -> Option<&mut SpaceByTime> {
        self.0.get_mut(&key)
    }

    #[inline]
    #[must_use]
    pub fn equal_keys(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .keys()
                .all(|key: &core::num::NonZero<u8>| other.0.contains_key(key))
    }

    #[inline]
    pub fn merge(&mut self, mut other: Self) {
        assert!(
            self.equal_keys(&other),
            "cmk SpaceTimeLayers have different keys"
        );

        for (key, mine) in self.iter_mut() {
            let theirs = other
                .0
                .remove(key)
                .expect("cmk SpaceTimeLayers have different keys");
            mine.merge(theirs);
        }
    }
}

impl core::ops::Index<NonZeroU8> for SpaceTimeLayers {
    type Output = SpaceByTime;
    #[inline]
    fn index(&self, index: NonZeroU8) -> &Self::Output {
        &self.0[&index]
    }
}

impl<'a> IntoIterator for &'a mut SpaceTimeLayers {
    type Item = (&'a NonZeroU8, &'a mut SpaceByTime);
    type IntoIter = std::collections::hash_map::IterMut<'a, NonZeroU8, SpaceByTime>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<'a> IntoIterator for &'a SpaceTimeLayers {
    type Item = (&'a NonZeroU8, &'a SpaceByTime);
    type IntoIter = std::collections::hash_map::Iter<'a, NonZeroU8, SpaceByTime>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl SpaceTimeLayers {
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&NonZeroU8, &SpaceByTime)> {
        self.0.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&NonZeroU8, &mut SpaceByTime)> {
        self.0.iter_mut()
    }
}
