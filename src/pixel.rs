use crate::{LANES_CMK, PixelPolicy};
use core::simd::prelude::*;
use derive_more::Display;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(transparent)]
#[derive(
    Debug,
    Default,
    Copy,
    Clone,
    IntoBytes,
    FromBytes,
    Immutable,
    Display,
    PartialEq,
    Eq,
    KnownLayout,
)]
pub struct Pixel(pub(crate) u8);

impl Pixel {
    pub(crate) const WHITE: Self = Self(0);
    const SPLAT_1: Simd<u8, LANES_CMK> = Simd::<u8, LANES_CMK>::splat(1);

    #[inline]
    pub(crate) fn slice_merge_with_white(pixels: &mut [Self]) {
        // Safety: Pixel is repr(transparent) around u8, so this cast is safe
        let bytes: &mut [u8] = pixels.as_mut_bytes();

        // Process with SIMD where possible
        let (prefix, chunks, suffix) = bytes.as_simd_mut::<LANES_CMK>();

        // Process SIMD chunks
        for chunk in chunks {
            *chunk >>= Self::SPLAT_1;
        }

        // Process remaining elements
        for byte in prefix.iter_mut().chain(suffix.iter_mut()) {
            *byte >>= 1;
        }
    }

    #[inline]
    pub(crate) fn slice_merge(left: &mut [Self], right: &[Self]) {
        debug_assert_eq!(
            left.len(),
            right.len(),
            "Both slices must have the same length"
        );

        let left_bytes: &mut [u8] = left.as_mut_bytes();
        let right_bytes: &[u8] = right.as_bytes();

        // Process chunks with SIMD where possible
        let (left_prefix, left_chunks, left_suffix) = left_bytes.as_simd_mut::<LANES_CMK>();
        let (right_prefix, right_chunks, right_suffix) = right_bytes.as_simd::<LANES_CMK>();

        // Process SIMD chunks using (a & b) + ((a ^ b) >> 1) formula
        for (left_chunk, right_chunk) in left_chunks.iter_mut().zip(right_chunks.iter()) {
            let a_and_b = *left_chunk & *right_chunk;
            *left_chunk ^= *right_chunk;
            *left_chunk >>= Self::SPLAT_1;
            *left_chunk += a_and_b;
        }

        assert!(left_prefix.is_empty() && right_prefix.is_empty());

        // Process remaining elements in suffix
        for (left_byte, right_byte) in left_suffix.iter_mut().zip(right_suffix.iter()) {
            *left_byte = (*left_byte & *right_byte) + ((*left_byte ^ *right_byte) >> 1);
        }
    }

    #[inline]
    pub(crate) const fn merge(&mut self, other: Self) {
        self.0 = (self.0 >> 1) + (other.0 >> 1) + ((self.0 & other.0) & 1);
    }

    #[inline]
    pub(crate) fn merge_slice_down_sample(
        slice: &[Self],
        empty_count: usize,
        pixel_policy: PixelPolicy,
    ) -> Self {
        match pixel_policy {
            PixelPolicy::Sampling => slice[0],
            PixelPolicy::Binning => {
                // cmk0000 make this faster with SIMD or at least more functional
                let mut sum: usize = 0;
                for i in 0..slice.len() {
                    sum += slice[i].0 as usize;
                }
                let total_len = crate::PowerOfTwo::from_usize_unchecked(slice.len() + empty_count);
                let mean = total_len.divide_into(sum) as u8;
                Self(mean)
            }
        }
    }

    #[inline]
    pub(crate) fn merge_slice_all(slice: &[Self], empty_count: i64) -> Self {
        let sum: u32 = slice.iter().map(|pixel: &Self| pixel.0 as u32).sum();
        let count = slice.len() + empty_count as usize;
        debug_assert!(count.is_power_of_two(), "Count must be a power of two");
        Self(crate::PowerOfTwo::from_u64_unchecked(count as u64).divide_into(sum) as u8)
    }
}

impl From<bool> for Pixel {
    #[inline]
    fn from(value: bool) -> Self {
        Self(value as u8 * 255)
    }
}

impl From<u8> for Pixel {
    #[inline]
    fn from(value: u8) -> Self {
        Self(value)
    }
}

impl From<&u8> for Pixel {
    #[inline]
    fn from(value: &u8) -> Self {
        Self(*value)
    }
}

impl From<u32> for Pixel {
    #[inline]
    fn from(value: u32) -> Self {
        debug_assert!(value <= 255, "Value must be less than or equal to 255");
        Self(value as u8)
    }
}

impl From<&u32> for Pixel {
    #[inline]
    fn from(value: &u32) -> Self {
        debug_assert!(*value <= 255, "Value must be less than or equal to 255");
        Self(*value as u8)
    }
}
