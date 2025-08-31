use core::num::NonZeroU8;

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
    KnownLayout,
    PartialEq,
    Eq,
)]
pub struct Symbol(u8);

impl Symbol {
    pub const STATE_ZERO: Self = Self(0);
    pub const STATE_ONE: Self = Self(1);
    pub const STATE_TWO: Self = Self(2);
    pub const STATE_THREE: Self = Self(3);
    pub const STATE_FOUR: Self = Self(4);
    pub const STATE_FIVE: Self = Self(5);
    pub const STATE_SIX: Self = Self(6);
    pub const STATE_SEVEN: Self = Self(7);
    pub const STATE_EIGHT: Self = Self(8);
    pub const STATE_NINE: Self = Self(9);
    // cmk what if more than 5 states?

    #[inline]
    pub const fn select_to_u32(self, select: NonZeroU8) -> u32 {
        (self.0 == select.get()) as u32 // Maps 0 → 0, select → 1
    }
}

impl From<Symbol> for u8 {
    #[inline]
    fn from(Symbol(symbol): Symbol) -> Self {
        symbol
    }
}

impl From<Symbol> for usize {
    fn from(Symbol(symbol): Symbol) -> Self {
        symbol as Self
    }
}

impl From<Symbol> for u32 {
    fn from(Symbol(symbol): Symbol) -> Self {
        symbol as Self
    }
}

// cmk00 remove support for this because they may be more than two states
impl From<bool> for Symbol {
    fn from(value: bool) -> Self {
        if value {
            Self::STATE_ONE
        } else {
            Self::STATE_ZERO
        }
    }
}
