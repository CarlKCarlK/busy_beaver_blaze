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
    // cmk what if more than 5 states?

    #[inline]
    pub const fn select_to_u32(self, select: u8) -> u32 {
        (self.0 == select) as u32 // Maps 0 → 0, select → 1
    }
}

// cmk00000000 all these
impl From<Symbol> for u8 {
    #[inline]
    fn from(Symbol(symbol_u8): Symbol) -> Self {
        symbol_u8
    }
}

impl From<Symbol> for usize {
    fn from(Symbol(symbol_u8): Symbol) -> Self {
        symbol_u8 as Self
    }
}

impl From<Symbol> for u32 {
    fn from(Symbol(symbol_u8): Symbol) -> Self {
        symbol_u8 as Self
    }
}

impl From<bool> for Symbol {
    fn from(bool_: bool) -> Self {
        Self(bool_ as u8) // Maps `false -> FALSE`, `true -> TRUE`
    }
}

impl From<&Symbol> for usize {
    fn from(Symbol(symbol): &Symbol) -> Self {
        *symbol as Self
    }
}
