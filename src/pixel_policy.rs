#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PixelPolicy {
    Sampling,
    Binning,
}

impl From<bool> for PixelPolicy {
    #[inline]
    fn from(is_binning: bool) -> Self {
        if is_binning {
            Self::Binning
        } else {
            Self::Sampling
        }
    }
}

impl From<PixelPolicy> for bool {
    #[inline]
    fn from(policy: PixelPolicy) -> Self {
        matches!(policy, PixelPolicy::Binning)
    }
}
