#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PowerOfTwo(u8);

impl core::ops::Div for PowerOfTwo {
    type Output = Self;

    /// Will always be at least 1.
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.0 >= rhs.0,
            "Divisor must be less than or equal to dividend"
        );
        self.saturating_div(rhs)
    }
}

impl core::ops::Mul<usize> for PowerOfTwo {
    type Output = usize;

    #[inline]
    fn mul(self, rhs: usize) -> Self::Output {
        // Multiply rhs by 2^(self.0)
        // This is equivalent to shifting rhs left by self.0 bits.
        rhs * (1usize << self.0)
    }
}

// cmk make the auto constructor so private that it can't be used w/ modules, so that new check is run.

impl PowerOfTwo {
    /// The smallest valid `Smoothness` value, representing `2^0 = 1`.
    pub const ONE: Self = Self(0);
    pub const TWO: Self = Self(1);
    pub const FOUR: Self = Self(2);
    pub const EIGHT: Self = Self(3);
    pub const SIXTEEN: Self = Self(4);
    pub const THIRTY_TWO: Self = Self(5);
    pub const SIXTY_FOUR: Self = Self(6);
    pub const MIN: Self = Self(0);
    pub const MAX: Self = Self(63);

    #[inline]
    #[must_use]
    pub fn offset_to_align(self, len: usize) -> usize {
        debug_assert!(
            (self.0 as u32) < usize::BITS,
            "Cannot shift left by self.0 = {} for usize::BITS = {}, which would overflow.",
            self.0,
            usize::BITS
        );
        len.wrapping_neg() & ((1 << self.0) - 1)
    }

    #[inline]
    #[must_use]
    pub const fn from_exp(value: u8) -> Self {
        debug_assert!(value <= Self::MAX.0, "Value must be 63 or less");
        Self(value)
    }

    #[inline]
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        1 << self.0
    }

    #[inline]
    #[must_use]
    pub const fn saturating_div(self, rhs: Self) -> Self {
        // Subtract exponents; if the subtrahend is larger, saturate to 0 aks One
        Self(self.0.saturating_sub(rhs.0))
    }

    #[inline]
    pub const fn assign_saturating_div_two(&mut self) {
        self.0 = self.0.saturating_sub(1);
    }

    #[inline]
    #[must_use]
    pub const fn double(self) -> Self {
        debug_assert!(self.0 < Self::MAX.0, "Value must be 63 or less");
        Self(self.0 + 1)
    }

    #[inline]
    #[must_use]
    pub fn as_usize(self) -> usize {
        let bits = core::mem::size_of::<usize>() * 8;
        debug_assert!(
            (self.0 as usize) < bits,
            "Exponent {} too large for usize ({} bits)",
            self.0,
            bits
        );
        1 << self.0
    }

    // from u64
    #[allow(clippy::missing_panics_doc)]
    #[inline]
    #[must_use]
    pub fn from_u64_unchecked(value: u64) -> Self {
        debug_assert!(value.is_power_of_two(), "Value must be a power of two");
        Self::from_exp(value.trailing_zeros() as u8)
    }

    #[inline]
    #[must_use]
    pub const fn from_usize_unchecked(value: usize) -> Self {
        debug_assert!(value.is_power_of_two(), "Value must be a power of two");
        Self::from_exp(value.trailing_zeros() as u8)
    }

    // from u64
    #[allow(clippy::missing_panics_doc)]
    #[inline]
    #[must_use]
    pub fn from_u64(value: u64) -> Self {
        assert!(value.is_power_of_two(), "Value must be a power of two");
        Self::from_exp(value.trailing_zeros() as u8)
    }

    #[inline]
    #[must_use]
    pub const fn from_usize(value: usize) -> Self {
        assert!(value.is_power_of_two(), "Value must be a power of two");
        Self::from_exp(value.trailing_zeros() as u8)
    }

    //     // #[inline]
    // #[must_use]
    // pub const fn from_usize_const(value: usize) -> Self {
    //     debug_assert!(value.is_power_of_two(), "Value must be a power of two");
    //     Self(value.trailing_zeros() as u8)
    // }

    #[inline]
    #[must_use]
    pub const fn log2(self) -> u8 {
        self.0
    }

    #[inline]
    pub fn rem_into_u64<T>(self, x: T) -> T
    where
        T: Copy
            + core::ops::BitAnd<Output = T>
            + core::ops::Sub<Output = T>
            + From<u64>
            + PartialEq,
    {
        x & (T::from(self.as_u64()) - T::from(1u64))
    }

    #[inline]
    pub fn rem_into_usize<T>(self, x: T) -> T
    where
        T: Copy
            + core::ops::BitAnd<Output = T>
            + core::ops::Sub<Output = T>
            + From<usize>
            + PartialEq,
    {
        x & (T::from(self.as_usize()) - T::from(1usize))
    }

    #[inline]
    #[must_use]
    pub fn rem_euclid_into(self, dividend: i64) -> i64 {
        let divisor = 1i64 << self.0; // Compute 2^n
        debug_assert!(divisor > 0, "divisor must be a power of two");
        let mask = divisor - 1;
        let remainder = dividend & mask;

        // If the remainder is negative, make it positive by adding divisor
        remainder + (divisor & (remainder >> Self::MAX.0))
    }

    #[inline]
    #[must_use]
    pub fn div_ceil_into<T>(self, other: T) -> T
    where
        T: Copy
            + core::ops::Add<Output = T>
            + core::ops::Sub<Output = T>
            + core::ops::Shl<u8, Output = T>
            + core::ops::Shr<u8, Output = T>
            + From<u8>,
    {
        let one = T::from(1);
        let two_pow = one << self.0;
        (other + two_pow - one) >> self.0
    }

    #[inline]
    pub fn divide_into<T>(self, x: T) -> T
    where
        T: Copy + core::ops::Shr<u8, Output = T>,
    {
        x >> self.0
    }

    #[inline]
    #[must_use]
    pub const fn divides_u64(self, x: u64) -> bool {
        // If x is divisible by 2^(self.0), shifting right then left recovers x.
        (x >> self.0) << self.0 == x
    }

    #[inline]
    #[must_use]
    pub const fn divides_i64(self, x: i64) -> bool {
        (x >> self.0) << self.0 == x
    }

    #[inline]
    #[must_use]
    pub const fn divides_usize(self, x: usize) -> bool {
        (x >> self.0) << self.0 == x
    }

    //     // #[inline]
    // #[must_use]
    // pub const fn divides_smoothness(self, other: Self) -> bool {
    //     self.0 <= other.0
    // }
}
