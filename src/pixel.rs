use crate::{LANES_CMK, PixelPolicy, bool_u8::BoolU8};
use core::ops::{Add, AddAssign};
use core::simd::{self, prelude::*};
use derive_more::Display;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// We define +, += to be the average of two pixels.
#[repr(transparent)]
#[derive(
    Default, Copy, Clone, IntoBytes, FromBytes, Immutable, Display, PartialEq, Eq, KnownLayout,
)]
pub struct Pixel(u8);

impl Add for Pixel {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self(Self::mean_bytes(self.0, other.0))
    }
}

impl AddAssign for Pixel {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        Self::mean_assign_bytes(&mut self.0, other.0);
    }
}

// define a debug trait for Pixel that looks like "P127"
impl core::fmt::Debug for Pixel {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "P{}", self.0)
    }
}

impl Pixel {
    pub const WHITE: Self = Self(0);
    const SPLAT_1: Simd<u8, LANES_CMK> = Simd::<u8, LANES_CMK>::splat(1);

    // cmk00000000 could make this unbiased by adding 1 before shift
    #[must_use]
    #[inline]
    pub const fn mean_bytes(a: u8, b: u8) -> u8 {
        (a & b) + ((a ^ b) >> 1)
    }

    // cmk00 inconsistent with the terms 'mean', 'average', 'merge', 'binning'
    #[inline]
    const fn mean_assign_bytes(a: &mut u8, b: u8) {
        let a_and_b = *a & b;
        *a ^= b;
        *a >>= 1;
        *a += a_and_b;
    }

    #[inline]
    pub(crate) const fn as_u8(self) -> u8 {
        self.0
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

        Self::slice_merge_bytes(left_bytes, right_bytes);
    }

    #[inline]
    fn simd_precondition<T, const LANES: usize>(left: &[T], right: &[T]) -> bool
    where
        T: simd::SimdElement,
        simd::LaneCount<LANES>: simd::SupportedLaneCount,
    {
        if left.len() != right.len() {
            return false;
        }

        let align = core::mem::align_of::<Simd<T, LANES>>();
        let left_align = left.as_ptr().align_offset(align);
        let right_align = right.as_ptr().align_offset(align);

        left_align == right_align
    }

    #[inline]
    pub(crate) fn slice_merge_bytes(left_bytes: &mut [u8], right_bytes: &[u8]) {
        // cmk debug_assert
        assert!(
            Self::simd_precondition::<u8, LANES_CMK>(left_bytes, right_bytes),
            "SIMD precondition failed"
        );
        // Process chunks with SIMD where possible
        let (left_prefix, left_chunks, left_suffix) = left_bytes.as_simd_mut::<LANES_CMK>();
        let (right_prefix, right_chunks, right_suffix) = right_bytes.as_simd::<LANES_CMK>();

        // Process prefix elements
        for (left_byte, right_byte) in left_prefix.iter_mut().zip(right_prefix.iter()) {
            Self::mean_assign_bytes(left_byte, *right_byte);
        }

        // cmk00000000 could make this unbiased by adding 1 before shift
        // Process SIMD chunks using (a & b) + ((a ^ b) >> 1) formula
        for (left_chunk, right_chunk) in left_chunks.iter_mut().zip(right_chunks.iter()) {
            let a_and_b = *left_chunk & *right_chunk;
            *left_chunk ^= *right_chunk;
            *left_chunk >>= Self::SPLAT_1;
            *left_chunk += a_and_b;
        }

        // Process remaining elements in suffix
        for (left_byte, right_byte) in left_suffix.iter_mut().zip(right_suffix.iter()) {
            Self::mean_assign_bytes(left_byte, *right_byte);
        }
    }

    // cmk000 see if this can be removed
    #[inline]
    pub(crate) fn slice_merge_bytes_no_simd(left_bytes: &mut [u8], right_bytes: &[u8]) {
        for (left_byte, right_byte) in left_bytes.iter_mut().zip(right_bytes.iter()) {
            Self::mean_assign_bytes(left_byte, *right_byte);
        }
    }

    #[inline]
    pub(crate) fn slice_merge_with_white(left: &mut [Self]) {
        let left_bytes: &mut [u8] = left.as_mut_bytes();

        // Process chunks with SIMD where possible
        let (left_prefix, left_chunks, left_suffix) = left_bytes.as_simd_mut::<LANES_CMK>();

        for left_chunk in left_chunks.iter_mut() {
            // divide by 2
            *left_chunk >>= Self::SPLAT_1;
        }

        assert!(left_prefix.is_empty());

        // Process remaining elements in suffix
        for left_byte in left_suffix.iter_mut() {
            // divide by 2
            *left_byte >>= 1;
        }
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
                // cmk000000 make this faster with SIMD or at least more functional
                let sum: u32 = slice.iter().map(|pixel| pixel.0 as u32).sum();
                let total_len = crate::PowerOfTwo::from_usize_unchecked(slice.len() + empty_count);
                let mean = total_len.divide_into(sum) as u8;
                Self(mean)
            }
        }
    }

    // cmk0000 This is only called from one place and empty_count is always 0. Is there already
    // cmk0000 a SIMD version of this?
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

impl From<Pixel> for u8 {
    #[inline]
    fn from(value: Pixel) -> Self {
        value.0
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

impl From<BoolU8> for Pixel {
    fn from(bool_u8: BoolU8) -> Self {
        Self(u8::from(bool_u8) * 255) // Maps 0 → 0, 1 → 255
    }
}
