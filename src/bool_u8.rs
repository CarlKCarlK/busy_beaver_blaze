use derive_more::Display;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(transparent)]
#[derive(Debug, Default, Copy, Clone, IntoBytes, FromBytes, Immutable, Display, KnownLayout)]
pub struct BoolU8(u8);

impl BoolU8 {
    pub const FALSE: Self = Self(0);
    pub const TRUE: Self = Self(1);
}

impl From<BoolU8> for u8 {
    #[inline]
    fn from(bool_u8: BoolU8) -> Self {
        bool_u8.0
    }
}

impl From<BoolU8> for usize {
    fn from(bool_u8: BoolU8) -> Self {
        bool_u8.0 as Self
    }
}

impl From<BoolU8> for u32 {
    fn from(bool_u8: BoolU8) -> Self {
        bool_u8.0 as Self
    }
}

impl From<bool> for BoolU8 {
    fn from(bool_: bool) -> Self {
        Self(bool_ as u8) // Maps `false -> FALSE`, `true -> TRUE`
    }
}

impl From<&BoolU8> for usize {
    fn from(bool_u8: &BoolU8) -> Self {
        bool_u8.0 as Self
    }
}

impl From<&BoolU8> for u32 {
    fn from(bool_u8: &BoolU8) -> Self {
        bool_u8.0 as Self
    }
}
