use crate::{ALIGN, symbol_u8::SymbolU8};
use aligned_vec::AVec;
use core::ops::{Add, AddAssign};
use core::simd::{self, prelude::*};
use derive_more::Display;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// We define +, += to be the average of two pixels, TODO is this wise?
#[repr(transparent)]
#[derive(
    Default, Copy, Clone, IntoBytes, FromBytes, Immutable, Display, PartialEq, Eq, KnownLayout,
)]
pub struct Pixel(u8);

impl Add for Pixel {
    type Output = Self;

    #[inline]
    fn add(self, Self(other): Self) -> Self {
        let Self(value) = self;
        Self(Self::mean_bytes(value, other))
    }
}

impl AddAssign for Pixel {
    #[inline]
    fn add_assign(&mut self, Self(other): Self) {
        let Self(value) = self;
        Self::mean_assign_bytes(value, other);
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
    const SPLAT_1: Simd<u8, ALIGN> = Simd::<u8, ALIGN>::splat(1);

    #[must_use]
    #[inline]
    pub const fn mean_bytes(first: u8, second: u8) -> u8 {
        (first & second) + ((first ^ second) >> 1)
    }

    // TODO inconsistent with the terms 'mean', 'average', 'merge', 'binning'
    #[inline]
    const fn mean_assign_bytes(first: &mut u8, second: u8) {
        let first_and_second = *first & second;
        *first ^= second;
        *first >>= 1;
        *first += first_and_second;
    }

    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        let Self(value) = self;
        value
    }

    #[inline]
    pub(crate) fn avec_merge_simd(left: &mut AVec<Self>, right: &AVec<Self>) {
        assert!(
            left.len() <= right.len(),
            "Left slice must be smaller than or equal to right slice"
        );
        left.resize(right.len(), Self::WHITE);
        Self::slice_merge_simd(left, right);
    }

    #[inline]
    pub(crate) fn slice_merge_simd(left: &mut [Self], right: &[Self]) {
        assert!(left.len() == right.len());

        let left_bytes: &mut [u8] = left.as_mut_bytes();
        let right_bytes: &[u8] = right.as_bytes();

        Self::slice_merge_bytes_simd(left_bytes, right_bytes);
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
    pub(crate) fn slice_merge_bytes_simd(left_bytes: &mut [u8], right_bytes: &[u8]) {
        // TODO Look at all asserts and think if some should be debug_assert.
        assert!(
            Self::simd_precondition::<u8, ALIGN>(left_bytes, right_bytes),
            "SIMD precondition failed"
        );
        // Process chunks with SIMD where possible
        let (left_prefix, left_chunks, left_suffix) = left_bytes.as_simd_mut::<ALIGN>();
        let (right_prefix, right_chunks, right_suffix) = right_bytes.as_simd::<ALIGN>();

        // Process prefix elements
        for (left_byte, right_byte) in left_prefix.iter_mut().zip(right_prefix.iter()) {
            Self::mean_assign_bytes(left_byte, *right_byte);
        }

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

    #[inline]
    pub(crate) fn slice_merge_bytes_no_simd(left_bytes: &mut [u8], right_bytes: &[u8]) {
        for (left_byte, right_byte) in left_bytes.iter_mut().zip(right_bytes.iter()) {
            Self::mean_assign_bytes(left_byte, *right_byte);
        }
    }

    #[inline]
    pub(crate) fn slice_merge_with_white_simd(left: &mut [Self]) {
        let left_bytes: &mut [u8] = left.as_mut_bytes();

        // Process chunks with SIMD where possible
        let (left_prefix, left_chunks, left_suffix) = left_bytes.as_simd_mut::<ALIGN>();

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
    fn from(Pixel(value): Pixel) -> Self {
        value
    }
}

impl From<&Pixel> for u8 {
    #[inline]
    fn from(Pixel(value): &Pixel) -> Self {
        *value
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

// cmk000000000
impl From<SymbolU8> for Pixel {
    fn from(symbol_u8: SymbolU8) -> Self {
        // todo!("Implement conversion from SymbolU8 to Pixel");
        Self(u8::from(symbol_u8) * 255) // Maps 0 → 0, 1 → 255
    }
}
