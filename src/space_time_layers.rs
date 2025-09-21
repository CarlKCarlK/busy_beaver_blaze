// cmk00 should this be space_by_time_layers???
extern crate alloc;
use alloc::collections::{BTreeMap, btree_map};
use core::num::NonZeroU8;

use crate::{Error, SpaceByTime, encode_png};
use aligned_vec::AVec;

/// Layers is an anonymous wrapper around `BTreeMap`<`NonZeroU8`, `SpaceByTime`>
#[derive(Clone, Default)]
pub struct SpaceTimeLayers(BTreeMap<NonZeroU8, SpaceByTime>);

impl SpaceTimeLayers {
    #[inline]
    pub fn first(&self) -> &SpaceByTime {
        assert!(!self.0.is_empty());
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
        debug_assert!(
            self.equal_keys(&other),
            "SpaceTimeLayers have different keys"
        );

        for (key, mine) in self.iter_mut() {
            let theirs = other
                .0
                .remove(key)
                .expect("SpaceTimeLayers have different keys");
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
    type IntoIter = btree_map::IterMut<'a, NonZeroU8, SpaceByTime>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<'a> IntoIterator for &'a SpaceTimeLayers {
    type Item = (&'a NonZeroU8, &'a SpaceByTime);
    type IntoIter = btree_map::Iter<'a, NonZeroU8, SpaceByTime>;
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

impl SpaceTimeLayers {
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn png_data_and_packed_data(
        &mut self,
        colors: &[[u8; 3]],
        tape_negative_len: usize,
        tape_nonnegative_len: usize,
        (goal_width, goal_height): (usize, usize),
    ) -> Result<(Vec<u8>, u32, u32, Vec<AVec<u8>>), Error> {
        let mut actual_width_height: Option<(u32, u32)> = None;
        let mut image_data_layers: Vec<AVec<u8>> = Vec::new();

        for (_, space_by_time) in self.iter_mut() {
            let (x_actual, y_actual, packed_data) = space_by_time.to_packed_data(
                tape_negative_len,
                tape_nonnegative_len,
                goal_width,
                goal_height,
            )?;

            if let Some((width, height)) = actual_width_height {
                assert!(
                    width == x_actual && height == y_actual,
                    "Layer dimensions must match"
                );
            } else {
                actual_width_height = Some((x_actual, y_actual));
            }
            image_data_layers.push(packed_data);
        }

        let (width, height) = actual_width_height
            .expect("No SpaceByTime layers in SpaceTimeLayers::collect_packed_data_with_dims");

        let png = encode_png(width, height, colors, image_data_layers.as_slice())
            .expect("Failed to encode PNG");

        Ok((png, width, height, image_data_layers))
    }
}
