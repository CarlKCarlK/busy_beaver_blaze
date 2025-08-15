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
pub struct SymbolU8(u8);

impl SymbolU8 {
    pub const STATE_ZERO: Self = Self(0);
    pub const STATE_ONE: Self = Self(1);
    pub const STATE_TWO: Self = Self(2);
    pub const STATE_THREE: Self = Self(3);
    pub const STATE_FOUR: Self = Self(4);
    // cmk what if more than 5 states?
}

impl From<SymbolU8> for u8 {
    #[inline]
    fn from(SymbolU8(symbol_u8): SymbolU8) -> Self {
        symbol_u8
    }
}

impl From<SymbolU8> for usize {
    fn from(SymbolU8(symbol_u8): SymbolU8) -> Self {
        symbol_u8 as Self
    }
}

impl From<SymbolU8> for u32 {
    fn from(SymbolU8(symbol_u8): SymbolU8) -> Self {
        symbol_u8 as Self
    }
}

impl From<bool> for SymbolU8 {
    fn from(bool_: bool) -> Self {
        Self(bool_ as u8) // Maps `false -> FALSE`, `true -> TRUE`
    }
}

impl From<&SymbolU8> for usize {
    fn from(SymbolU8(symbol_u8): &SymbolU8) -> Self {
        *symbol_u8 as Self
    }
}

impl From<&SymbolU8> for u32 {
    fn from(SymbolU8(symbol_u8): &SymbolU8) -> Self {
        *symbol_u8 as Self
    }
}
